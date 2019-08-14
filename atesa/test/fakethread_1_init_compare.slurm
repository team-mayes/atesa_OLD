#!/bin/bash
#SBATCH --job-name="fakethread_1_init"
#SBATCH --output="fakethread_1_init.%j.%N.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 01:00:00

set -x
module load amber
module load python
export PYTHONPATH=/opt/amber/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/amber/lib:$LD_LIBRARY_PATH
export AMBERHOME=/share/apps/compute/amber16.dat

ibrun /opt/amber/16/bin/sander.MPI -ng 1 -groupfile none -O -i /Users/tburgin/Documents/PycharmProjects/atesa/atesa/test/input_files/init.in -o fakethread_1_init.out -p test.prmtop -c fakethread -r fakethread_1_init_fwd.rst -x fakethread_1_init.nc