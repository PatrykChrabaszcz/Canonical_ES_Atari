from src.optimizers import OpenAIOptimizer, CanonicalESOptimizer, CanonicalESMeanOptimizer
from src.policy import Policy
from src.logger import Logger

from argparse import ArgumentParser
from mpi4py import MPI
import numpy as np
import time
import json
import gym


# This will allow us to create optimizer based on the string value from the configuration file.
# Add you optimizers to this dictionary.
optimizer_dict = {
    'OpenAIOptimizer': OpenAIOptimizer,
    'CanonicalESOptimizer': CanonicalESOptimizer,
    'CanonicalESMeanOptimizer': CanonicalESMeanOptimizer
}


# Main function that executes training loop.
# Population size is derived from the number of CPUs
# and the number of episodes per CPU.
# One CPU (id: 0) is used to evaluate currently proposed
# solution in each iteration.
# run_name comes useful when the same hyperparameters
# are evaluated multiple times.
def main(ep_per_cpu, game, configuration_file, run_name):
    start_time = time.time()

    with open(configuration_file, 'r') as f:
        configuration = json.loads(f.read())

    env_name = '%sNoFrameskip-v4' % game

    # MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    # One cpu (rank 0) will evaluate results
    train_cpus = cpus - 1

    # Deduce population size
    lam = train_cpus * ep_per_cpu

    # Create environment
    env = gym.make(env_name)

    # Create policy (Deep Neural Network)
    # Internally it applies preprocessing to the environment state
    policy = Policy(env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])

    # Create reference batch used for normalization
    # It will be overwritten with vb from worker with rank 0
    vb = policy.get_vb()

    # Extract vector with current parameters.
    parameters = policy.get_parameters()

    # Send parameters from worker 0 to all workers (MPI stuff)
    # to ensure that every worker starts in the same position
    comm.Bcast([parameters, MPI.FLOAT], root=0)
    comm.Bcast([vb, MPI.FLOAT], root=0)

    # Set the same virtual batch for each worker
    if rank != 0:
        policy.set_vb(vb)

    # Create optimizer with user defined settings (hyperparameters)
    OptimizerClass = optimizer_dict[configuration['optimizer']]
    optimizer = OptimizerClass(parameters, lam, rank, configuration["settings"])

    # Only rank 0 worker will log information from the training
    logger = None
    if rank == 0:
        # Initialize logger, save virtual batch and save some basic stuff at the beginning
        logger = Logger(optimizer.log_path(game, configuration['network'], run_name))
        logger.save_vb(vb)

        # Log basic stuff
        logger.log('Game'.ljust(25) + '%s' % game)
        logger.log('Network'.ljust(25) + '%s' % configuration['network'])
        logger.log('Optimizer'.ljust(25) + '%s' % configuration['optimizer'])
        logger.log('Number of CPUs'.ljust(25) + '%d' % cpus)
        logger.log('Population'.ljust(25) + '%d' % lam)
        logger.log('Dimensionality'.ljust(25) + '%d' % len(parameters))

        # Log basic info from the optimizer
        optimizer.log_basic(logger)

    # We will count number of steps
    # frames = 4 * steps (3 * steps for SpaceInvaders)
    steps_passed = 0
    while True:
        # Iteration start time
        iter_start_time = time.time()
        # Workers that run train episodes
        if rank != 0:
            # Empty arrays for each episode. We save: length, reward, noise index
            lens = [0] * ep_per_cpu
            rews = [0] * ep_per_cpu
            inds = [0] * ep_per_cpu

            # For each episode in this CPU we get new parameters,
            # update policy network and perform policy rollout
            for i in range(ep_per_cpu):
                ind, p = optimizer.get_parameters()
                policy.set_parameters(p)
                e_rew, e_len = policy.rollout()
                lens[i] = e_len
                rews[i] = e_rew
                inds[i] = ind

            # Aggregate information, will later send it to each worker using MPI
            msg = np.array(rews + lens + inds, dtype=np.int32)

        # Worker rank 0 that runs evaluation episodes
        else:
            rews = [0] * ep_per_cpu
            lens = [0] * ep_per_cpu
            for i in range(ep_per_cpu):
                ind, p = optimizer.get_parameters()
                policy.set_parameters(p)
                e_rew, e_len = policy.rollout()
                rews[i] = e_rew
                lens[i] = e_len

            eval_mean_rew = np.mean(rews)
            eval_max_rew = np.max(rews)

            # Empty array, evaluation results are not used for the update
            msg = np.zeros(3 * ep_per_cpu, dtype=np.int32)

        # MPI stuff
        # Initialize array which will be updated with information from all workers using MPI
        results = np.empty((cpus, 3 * ep_per_cpu), dtype=np.int32)
        comm.Allgather([msg, MPI.INT], [results, MPI.INT])

        # Skip empty evaluation results from worker with id 0
        results = results[1:, :]

        # Extract IDs and rewards
        rews = results[:, :ep_per_cpu].flatten()
        lens = results[:, ep_per_cpu:(2*ep_per_cpu)].flatten()
        ids = results[:, (2*ep_per_cpu):].flatten()

        # Update parameters
        optimizer.update(ids=ids, rewards=rews)

        # Steps passed = Sum of episode steps from all offsprings
        steps = np.sum(lens)
        steps_passed += steps

        # Write some logs for this iteration
        # Using logs we are able to recover solution saved
        # after 1 hour of training or after 1 billion frames
        if rank == 0:
            iteration_time = (time.time() - iter_start_time)
            time_elapsed = (time.time() - start_time)/60
            train_mean_rew = np.mean(rews)
            train_max_rew = np.max(rews)
            logger.log('------------------------------------')
            logger.log('Iteration'.ljust(25) + '%f' % optimizer.iteration)
            logger.log('EvalMeanReward'.ljust(25) + '%f' % eval_mean_rew)
            logger.log('EvalMaxReward'.ljust(25) + '%f' % eval_max_rew)
            logger.log('TrainMeanReward'.ljust(25) + '%f' % train_mean_rew)
            logger.log('TrainMaxReward'.ljust(25) + '%f' % train_max_rew)
            logger.log('StepsSinceStart'.ljust(25) + '%f' % steps_passed)
            logger.log('StepsThisIter'.ljust(25) + '%f' % steps)
            logger.log('IterationTime'.ljust(25) + '%f' % iteration_time)
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed)

            # Give optimizer a chance to log its own stuff
            optimizer.log(logger)
            logger.log('------------------------------------')

            # Write stuff for training curve plot
            stat_string = "{},\t{},\t{},\t{},\t{},\t{}\n".\
                format(steps_passed, (time.time()-start_time),
                       eval_mean_rew, eval_max_rew, train_mean_rew, train_max_rew)
            logger.write_general_stat(stat_string)
            logger.write_optimizer_stat(optimizer.stat_string())

            # Save currently proposed solution every 20 iterations
            if optimizer.iteration % 20 == 1:
                logger.save_parameters(optimizer.parameters, optimizer.iteration)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-e', '--episodes_per_cpu',
                        help="Number of episode evaluations for each CPU, "
                             "population_size = episodes_per_cpu * Number of CPUs",
                        default=1, type=int)
    parser.add_argument('-g', '--game', help="Atari Game used to train an agent")
    parser.add_argument('-c', '--configuration_file', help='Path to configuration file')
    parser.add_argument('-r', '--run_name', help='Name of the run, used to create log folder name', type=str)
    args = parser.parse_args()
    return args.episodes_per_cpu, args.game, args.configuration_file, args.run_name


if __name__ == '__main__':
    ep_per_cpu, game, configuration_file, run_name = parse_arguments()
    main(ep_per_cpu, game, configuration_file, run_name)
