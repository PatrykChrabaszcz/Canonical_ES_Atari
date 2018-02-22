for game in  Pong # SpaceInvaders Enduro BeamRider Breakout Qbert Seaquest Alien
do
    for run in 1 2 3
    do
        msub -v EPISODES_PER_CPU=2,GAME=$game,CONFIG="configurations/sample_configuration.json",RUN=$run -l nodes=5:ppn=20,walltime=10:00:00 submit.sh
    done
done