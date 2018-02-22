import numpy as np
from itertools import groupby
from operator import itemgetter


class BaseOptimizer(object):
    def __init__(self, parameters, rank):
        # Worker id (MPI stuff).
        self.rank = rank
        # 2 GB of random noise as in OpenAI paper.
        self.noise_table = np.random.RandomState(123).randn(int(5e8)).astype('float32')
        # Dimensionality of the problem
        self.n = len(parameters)
        # Current solution (The one that we report).
        self.parameters = parameters
        # Computed update, step in parameter space computed in each iteration.
        self.step = 0

        # Should be increased when iteration is done.
        self.iteration = 0

    # Sample random index from the noise table.
    def r_noise_id(self):
        return np.random.random_integers(0, len(self.noise_table)-self.n)

    # Returns parameters to evaluate for a worker and an ID.
    # ID is used when computing update step (It might indicate index in the noise table).
    def get_parameters(self):
        raise NotImplementedError

    # Function to compute how far from the origin the current solution is and how
    # big are the update steps.
    def magnitude(self, vec):
        return np.sqrt(np.sum(np.square(vec)))

    # Updates Optimizer based on IDs and rewards from evaluations
    def update(self, ids, rewards):
        raise NotImplementedError

    # Use logger to log basic info after each iteration
    def log(self, logger):
        raise NotImplementedError

    # Use logger to log basic info.
    # Will be called at the beginning of the training.
    def log_basic(self, logger):
        raise NotImplementedError

    # Each optimizer might have different folder structure to log results.
    # Advice: derive log_path from optimizer parameters and this function
    # parameters.
    def log_path(self, game, network, run_name):
        raise NotImplementedError

    # Used to log optimizer specific statistics.
    # Will be called after each iteration and saved in separate file.
    def stat_string(self):
        return None


class OpenAIOptimizer(BaseOptimizer):
    # Place for OpenAI algorithm from:
    # Evolution Strategies as a Scalable Alternative to Reinforcement Learning
    # https://arxiv.org/pdf/1703.03864.pdf
    def __init__(self, parameters, lam, rank, settings):
        super().__init__(parameters, rank)

        self.lam = lam

        # Extract parameters from configuration file.
        self.sigma = settings['sigma']
        self.weight_decay = settings['weight_decay']
        self.lr = settings['learning_rate']
        self.momentum = settings['momentum']

        # Momentum.
        self.v = np.zeros_like(parameters)

        # Gradient.
        self.g = np.zeros_like(parameters)

        # Weight decay.
        self.wd = np.zeros_like(parameters)

        # Variables required to alternate between positive and negative noise (Mirror sampling).
        self.state = True
        self.r_id = None

    def get_parameters(self):
        # Worker 0 will always evaluate currently proposed solution.
        if self.rank == 0:
            return None, self.parameters

        # This will alter between returning parameters with positive and negative noise.
        else:
            if self.state:
                self.r_id = self.r_noise_id()
                p = self.parameters + self.sigma * self.noise_table[self.r_id:(self.r_id + self.n)]
            else:
                p = self.parameters - self.sigma * self.noise_table[self.r_id:(self.r_id + self.n)]

            self.state = not self.state

            return self.r_id, p

    # Noise index of evaluated solution is used as an ID in this optimizer.
    def update(self, ids, rewards):
        raise NotImplementedError('OpenAI implementation is not publicly available.')

    def log_basic(self, logger):
        logger.log('Lambda'.ljust(25) + '%d' % self.lam)
        logger.log('Sigma'.ljust(25) + '%f' % self.sigma)
        logger.log('WeightDecay'.ljust(25) + '%f' % self.weight_decay)
        logger.log('LearningRate'.ljust(25) + '%f' % self.lr)
        logger.log('Momentum'.ljust(25) + '%f' % self.momentum)
        logger.log('ParamNorm'.ljust(25) + '%f' % self.magnitude(self.parameters))

    def log(self, logger):
        logger.log('ParamNorm'.ljust(25) + '%f' % self.magnitude(self.parameters))
        logger.log('GradNorm'.ljust(25) + '%f' % self.magnitude(self.g))
        logger.log('WeightDecayNorm'.ljust(25) + '%f' % self.magnitude(self.wd))
        logger.log('StepNorm'.ljust(25) + '%f' % self.magnitude(self.step))

    def log_path(self, game, network, run_name):
        return "logs_mpi/%s/OpenAI/%s/%d/%f/%f/%f/%f/%s" % \
               (game, network, self.lam, self.sigma, self.lr, self.weight_decay, self.momentum, run_name)


class CanonicalESOptimizer(BaseOptimizer):
    # CanonicalES algorithm as in the paper:
    # Back to Basics: Benchmarking Canonical Evolution Strategies for Playing Atari
    def __init__(self, parameters, lam, rank, settings):
        super().__init__(parameters, rank)

        self.lam = lam
        self.sigma = settings['sigma']

        # One could experiment with different learning_rates.
        # Disabled for our experiments (by setting to 1).
        self.lr = settings['learning_rate']

        # Parent population size
        self.u = settings['mu']

        # One could experiment rescaling c_sigma to stronger adjust sample distribution noise.
        # Disabled for our experiments (by setting to 1).
        self.c_sigma_factor = settings['c_sigma_factor']

        assert(self.u <= self.lam)

        # Compute weights for weighted mean of the top self.u offsprings
        # (parents for the next generation).
        self.w = np.array([np.log(self.u + 0.5) - np.log(i) for i in range(1, self.u + 1)])
        self.w /= np.sum(self.w)

        # Noise adaptation stuff.
        self.p_sigma = np.zeros(self.n)
        self.u_w = 1 / float(np.sum(np.square(self.w)))
        self.c_sigma = (self.u_w + 2) / (self.n + self.u_w + 5)
        self.c_sigma *= self.c_sigma_factor
        self.const_1 = np.sqrt(self.u_w * self.c_sigma * (2 - self.c_sigma))

    def get_parameters(self):
        if self.rank == 0:
            return None, self.parameters
        r_id = self.r_noise_id()
        p = self.parameters + self.sigma * self.noise_table[r_id:(r_id + self.n)]
        return r_id, p

    def update(self, ids, rewards):
        # Best will point to solutions with the highest rewards
        # best[0] = index of the solution with the best reward
        best = np.array(rewards).argsort()[::-1][:self.u]
        step = np.zeros(self.n)

        # Simple weighted mean of top self.u offsprings
        for i in range(self.u):
            ind = ids[best[i]]
            step += self.w[i] * self.noise_table[ind:ind + self.n]

        self.step = self.lr * self.sigma * step
        self.parameters += self.step

        # Noise adaptation stuff.
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + self.const_1 * step
        self.sigma = self.sigma * np.exp((self.c_sigma / 2) * (np.sum(np.square(self.p_sigma)) / self.n - 1))

        self.iteration += 1

    def log_basic(self, logger):
        logger.log('Lambda'.ljust(25) + '%d' % self.lam)
        logger.log('Mu'.ljust(25) + '%d' % self.u)
        logger.log('LearningRate'.ljust(25) + '%f' % self.lr)
        logger.log('MuW'.ljust(25) + '%f' % self.u_w)
        logger.log('CSigma'.ljust(25) + '%f' % self.c_sigma)
        logger.log('CSigmaFactor'.ljust(25) + '%f' % self.c_sigma_factor)
        logger.log('Const1'.ljust(25) + '%f' % self.const_1)

    def log(self, logger):
        logger.log('ParamNorm'.ljust(20) + '%f' % self.magnitude(self.parameters))
        logger.log('StepNorm'.ljust(20) + '%f' % self.magnitude(self.step))
        logger.log('Sigma'.ljust(20) + '%f' % self.sigma)
        logger.log('PSigmaNorm'.ljust(20) + '%f' % self.magnitude(self.p_sigma))

    def log_path(self, game, network, run_name):
        return "logs_mpi/%s/Baseline/%s/%d/%d/%f/%f/%f/%s" % \
               (game, network, self.lam, self.u, self.sigma, self.lr, self.c_sigma_factor, run_name)

    # We might want to log other stuff as well ???
    def stat_string(self):
        str = '%g\n' % self.sigma
        return str


# Optimizer that will try to attenuate the effect of evaluation noise.
# No clear improvements noticed for some simple experiments we did.
class CanonicalESMeanOptimizer(CanonicalESOptimizer):
    def __init__(self, parameters, lam, rank, settings):
        eval_num = settings['eval_num']
        assert(eval_num >= 1)
        super().__init__(parameters, lam//eval_num, rank, settings)
        # Worker 0 has to be an evaluation worker
        if rank == 0:
            self.rank = 0
        else:
            self.rank = rank//eval_num + 1
        # We need the same random samples for a set of workers with the same rank
        self.random_state = np.random.RandomState(100 + self.rank)

    # We have to redefine this method
    # This will allow us to generate the same random numbers on different workers
    def r_noise_id(self):
        return self.random_state.random_integers(0, len(self.noise_table) - self.n)

    def update(self, ids, rewards):

        indices_mean = []
        rewards_mean = []
        for i, r in groupby(zip(ids, rewards), itemgetter(0)):
            rews = list(list(zip(*r))[1])
            mean_rew = sum(rews)/len(rews)
            indices_mean.append(i)
            rewards_mean.append(mean_rew)

        super().update(indices_mean, rewards_mean)

    def log_path(self, game, network, run_name):
        return "logs_mpi/%s/BaselineMean/%s/%d/%d/%f/%f/%f/%s" % \
               (game, network, self.lam, self.u, self.sigma, self.lr, self.c_sigma_factor, run_name)
