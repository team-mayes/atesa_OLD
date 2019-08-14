#!/bin/bash
#SBATCH --job-name="fakethread_ca_1"
#SBATCH --output="fakethread_ca_1.%j.%N.out"
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

ibrun /opt/amber/16/bin/sander.MPI -ng 1 -groupfile none -O -i /Users/tburgin/Documents/PycharmProjects/atesa/atesa/test/input_files/committor_analysis.in -o fakethread_ca_1.out -p ../test.prmtop -c ../fakethread -r fakethread_ca_1.rst -x fakethread_ca_1.nc