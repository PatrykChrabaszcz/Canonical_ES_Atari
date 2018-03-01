#!/bin/bash
module load mpi/openmpi/2.1-gnu-4.8
module load devel/python/3.6.0
source ~/venv/bin/activate

export LC_ALL=de_DE.utf-8
export LANG=de_DE.utf-8

cd $EVOLUTION
cd Canonical_ES_Atari

echo $EPISODES_PER_CPU
echo $GAME
echo $RUN
echo $CONFIG

mpirun --bind-to core --map-by node -report-bindings python main.py -e $EPISODES_PER_CPU -g $GAME -c $CONFIG -r $RUN
