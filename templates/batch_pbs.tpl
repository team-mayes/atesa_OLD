### PBS preamble

#PBS -N {{ name }}
#PBS -M tburgin@umich.edu
#PBS -m ae

#PBS -A hbmayes_fluxod
#PBS -l qos=flux
#PBS -q fluxod

#PBS -l nodes={{ nodes }}:ppn={{ taskspernode }}
#PBS -l walltime={{ walltime }}
#PBS -j oe
#PBS -V

### End PBS preamble

module load openmpi/1.10.2/gcc/4.8.5
module load gcc/4.8.5
module load cuda/7.5
module load amber

mpirun {{ solver }}.MPI -ng 1 -groupfile none -O -i {{ inp }} -o {{ out }} -p {{ prmtop }} -c {{ inpcrd }} -r {{ rst }} -x {{ nc }}
