 #! /usr/bin/env python

import os
import subprocess
import sys
import re
import time
import shutil
import pytraj
import random
import fileinput
import glob
import argparse
import ast

# Begin by adding to PYTHONPATH the desired location of the Jinja2 installation and installing it.
# os.environ['PYTHONPATH'] = os.path.dirname(os.path.realpath(sys.argv[0])) + '/python_packages/:$PYTHONPATH'
# subprocess.call('easy_install -d ' + os.path.dirname(os.path.realpath(sys.argv[0])) + '/python_packages/ -i https://pypi.python.org/simple Jinja2', shell=True)

# The above two lines are broken (on Comet, at least) for reasons having to do with permissions I think. In the end, the
# real way to handle this is to include Jinja as a dependency in the setup script, once I've built that. For now, I need
# to issue the following two commands from the aimless_shooting directory before testing:
#   PYTHONPATH=/home/tburgin/glycosynthases/171110_TmAfc_d224g/aimless_shooting/python_packages/:/home/tburgin/glycosynthases/171110_TmAfc_d224g/aimless_shooting/python_packages/lib/python2.7/site-packages/:$PYTHONPATH
#   easy_install -d ./python_packages -i https://pypi.python.org/simple Jinja2
# todo: build a proper setup.py (way down the line)

from jinja2 import Environment, FileSystemLoader

parser = argparse.ArgumentParser(description='Perform aimless shooting according to the settings given in the input file.')
parser.add_argument('-i', metavar='input_file', type=str, nargs=1, default='as.in', help='input filename; see documentation for format. Default=\'as.in\'')
arguments = vars(parser.parse_args())   # Retrieves arguments as a dictionary object

path = arguments.get('i')   # parser drops dashes ('-') from variable names, so the -i argument is retrieved as such
if type(path) == list:      # handles inconsistency in format when the default value is used vs. when a value is given
    path = path[0]
if not os.path.exists(path):
    sys.exit('Error: cannot find input file \'' + path + '\'')

input_file = open(path)
input_file_lines = [i.strip('\n').split(' ') for i in input_file.readlines()]

# Initialize default values for all the input file entries, to be overwritten by the actual contents of the file
initial_structure = 'inpcrd'                    # Initial structure filename
if_glob = False                                 # True if initial_structure should be interpreted as a glob argument
topology = 'prmtop'                             # Topology filename
n_adjust = 50                                   # Max number of frames by which each step deviates from the previous one
batch_system = 'slurm'                          # Batch system type, Slurm or PBS
working_directory = os.getcwd() + '/as_working' # Working directory for aimless shooting calculations
restart_on_crash = False                        # If a thread crashes, should it be resubmitted or allowed to terminate? #todo: either extend this behavior to type='prod' jobs ending with commit type 'fail', or be explicit that that's not what this does.

# Read in variables from the input file; there's probably a cleaner way to do this, but no matter.
for entry in input_file_lines:
    if entry[0] == 'initial_structure':
        initial_structure = entry[2]
    elif entry[0] == 'if_glob':
        if_glob = bool(entry[2])
    elif entry[0] == 'topology':
        topology = entry[2]
    elif entry[0] == 'n_adjust':
        n_adjust = int(entry[2])
    elif entry[0] == 'batch_system':
        batch_system = entry[2]
    elif entry[0] == 'working_directory':
        working_directory = os.getcwd() + '/' + entry[2]
    elif entry[0] == 'commit_fwd':
        commit_define_fwd = ast.literal_eval(entry[2])
    elif entry[0] == 'commit_bwd':
        commit_define_bwd = ast.literal_eval(entry[2])
    elif entry[0] == 'candidate_op':
        candidateops = ast.literal_eval(entry[2])
    elif entry[0] == 'restart_on_crash':
        restart_on_crash = bool(entry[2])

# Return an error and exit if the input file is missing entries for non-optional inputs.
if 'commit_fwd' not in [entry[0] for entry in input_file_lines]:
    sys.exit('Error: Input file is missing entry for commit_fwd, which is non-optional')
if 'commit_bwd' not in [entry[0] for entry in input_file_lines]:
    sys.exit('Error: Input file is missing entry for commit_bwd, which is non-optional')
if 'candidate_op' not in [entry[0] for entry in input_file_lines]:
    sys.exit('Error: Input file is missing entry for candidate_op, which is non-optional')

if if_glob == True:
    start_name = glob.glob(initial_structure) # list of names of Amber-interpretable coordinate files to begin shooting from
else:
    start_name = [initial_structure]

# todo: add a degeneracy option that uses the same start_name as seed for N different threads by duplicating the start_name file and appending a different number to each duplicate, and then adding each duplicate to the start_name list.

# Objects to define commitment.
# This data is formatted like [[mask1,mask2,...],[mask3,mask4,...],[distance mask1-to-mask3,distance mask2-to-mask4,...],[less than or greater than for 1-to-3,less than or greater than for 2-to-4]].
# Distances are in angstroms.
# todo: if a given mask matches more than one atom, the center of mass of that mask should be used.
# commit_define_fwd = [['@7185','@7185'],['@7174','@7186'],[1.60,2.75],['lt','gt']]
# commit_define_bwd = [['@7185','@7185'],['@7174','@7186'],[2.75,2.00],['gt','lt']]

# Objects to define candidate OPs.
# Like the commit_define objects, this is a index-paired list of lists. The distance between each pair of atoms given by
# the indices in a given pair is taken as a candidate order parameter to be measured and stored at the end of each
# successful shooting run. todo: add support for OPs other than distances, e.g., angles, dihedrals
# candidateops = [['@4273','@7175','@7174','@7185'],
#                 ['@7175','@7174','@7185','@7186']]

# Make a working directory
dirName = working_directory
if os.path.exists(dirName):
    shutil.rmtree(dirName)  # delete old working directory
os.makedirs(dirName)        # make a new one

# Initialize jinja2 environment for filling out templates
env = Environment(
    loader=FileSystemLoader(os.path.dirname(os.path.realpath(sys.argv[0])) + '/' + 'templates'),
)

os.chdir(os.getcwd() + '/' + 'as_working')  # move to working directory


# Define the "thread" object that constitutes one string of shooting moves in our search through phase space. Each
# thread is an independent series of simulations with its own acceptance ratio.
class Thread:
    basename = ''   # first part of name, universal to all files in the thread
    name = ''       # full name, identical to basename unless otherwise specified
    suffix = ''     # appended to basename after an underscore to build name. Should be str(an integer)
    jobid1 = ''     # current batch system jobid associated with the thread init step or forward trajectory
    jobid2 = ''     # another jobid slot, for the backward trajectory
    type = ''       # thread job type; either init to get initial trajectories or prod to run forward and backward
    start_name = '' # name of previous thread to initialize shooting from
    last_valid = '' # suffix corresponding to the most recent shooting point that resulted in a valid transition path
    commit1 = ''    # flag indicating commitment direction for the fwd trajectory
    commit2 = ''    # flag indicating commitment direction for the bwd trajectory
    accept_moves = 0# running total of completed shooting moves that were accepted
    total_moves = 0 # running total of completed shooting moves with any result
    prmtop = ''     # name of parameter/topology file corresponding to this thread


# Then, define a series of functions to perform the basic tasks of creating batch files, interacting with the batch
# system, and interpreting/modifying coordinate and trajectory files.
def makebatch(thread):
    # Makes batch files for the three jobs that constitute one shooting move

    name = thread.name
    type = thread.type
    batch = 'batch_' + batch_system + '_18.tpl'

    if type == 'init':
        # init batch file
        open('as.log', 'a').write('\nWriting init batch file for ' + name + ' starting from ' + thread.start_name)
        template = env.get_template(batch)
        filled = template.render(name=name + '_init', nodes='1', taskspernode='1', walltime='01:00:00', solver='sander',
                                 inp='../input_files/init.in', out=name + '_init.out', prmtop=thread.prmtop,
                                 inpcrd=thread.start_name, rst=name + '_init_fwd.rst', nc=name + '_init.nc', mem='4000mb')
        with open(name + '_init.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

    elif type == 'prod':
        # forward and backward simulation batch files
        open('as.log', 'a').write('\nWriting forward batch file for ' + name)
        template = env.get_template(batch)
        filled = template.render(name=name + '_fwd', nodes='1', taskspernode='1', walltime='03:00:00', solver='sander',
                                 inp='../input_files/prod.in', out=name + '_fwd.out', prmtop=thread.prmtop,
                                 inpcrd=name + '_init_fwd.rst', rst=name + '_fwd.rst', nc=name + '_fwd.nc', mem='4000mb')
        with open(name + '_fwd.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

        open('as.log', 'a').write('\nWriting backward batch file for ' + name)
        template = env.get_template(batch)
        filled = template.render(name=name + '_bwd', nodes='1', taskspernode='1', walltime='03:00:00', solver='sander',
                                 inp='../input_files/prod.in', out=name + '_bwd.out', prmtop=thread.prmtop,
                                 inpcrd=name + '_init_bwd.rst', rst=name + '_bwd.rst', nc=name + '_bwd.nc', mem='4000mb')
        with open(name + '_bwd.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

    else:
        open('as.log', 'a').write('\nAn incorrect job type \"' + type + '\" was passed to makebatch.')
        sys.exit('An incorrect job type \"' + type + '\" was passed to makebatch.')


def subbatch(thread,direction = ''):
    # Submits a batch file given by batch and returns its process number as a string
    name = thread.name  # just shorthand

    if direction == 'fwd':
        type = 'fwd'
    elif direction == 'bwd':
        type = 'bwd'
    else:
        type = thread.type

    if batch_system == 'pbs':
        command = 'qsub ' + name + '_' + type + '.' + batch_system
    elif batch_system == 'slurm':
        command = 'sbatch ' + name + '_' + type + '.' + batch_system
    else:
        open('as.log', 'a').write('An incorrect batch system type \"' + batch_system + '\" was passed to subbatch.')
        sys.exit('An incorrect batch system type \"' + batch_system + '\" was supplied. Acceptable types are: pbs, slurm')

    open('as.log', 'a').write('\nSubmitting job: ' + name + '_' + type + '.' + batch_system)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               close_fds=True, shell=True)
    output = process.stdout.read()
    while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):  # todo: fix this kludge
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read()
    open('as.log', 'a').write('\nBatch system says: ' + str(output))
    pattern = re.compile('[0-9]+')
    return pattern.findall(str(output))[0]


def spawnthread(basename, type = 'init', suffix=''):
    # Spawns a new thread object with the given arguments as the corresponding parameters. The name parameter is
    # constructed from the basename and suffix parameters.
    new_thread = Thread()
    new_thread.basename = basename
    new_thread.suffix = suffix
    new_thread.name = basename + '_' + suffix
    new_thread.type = type
    new_thread.start_name = basename
    return new_thread


def checkcommit(thread,direction):
    # Function to check for commitment of the given thread in the given direction using pytraj. Direction should be a
    # string, either 'bwd' or 'fwd'. The definitions of commitment are given by the two "commit_define" objects supplied
    # by the user.

    if not os.path.isfile(thread.name + '_' + direction + '.nc'):   # if the file doesn't exist yet, just do nothing
        return ''

    traj = pytraj.iterload(thread.name + '_' + direction + '.nc', thread.prmtop, format='.nc')

    if not traj:                        # catches error if the trajectory file exists but has zero frames
        return ''                       # would be nice to figure out how to suppress the pytraj Internal Error...

    commit_flag = ''                    # initialize flag for commitment. This is the value to be returned eventually.
    for i in range(0,len(commit_define_fwd[2])):
        if commit_define_fwd[3][i] == 'lt':
            if pytraj.distance(traj, commit_define_fwd[0][i] + ' ' + commit_define_fwd[1][i], n_frames=1)[-1] <= commit_define_fwd[2][i]:
                commit_flag = 'fwd'     # if a commitor test is passed, testing moves on to the next one.
            else:
                commit_flag = ''
                break                   # if a commitor test is not passed, all testing in this direction ends in a fail
        elif commit_define_fwd[3][i] == 'gt':
            if pytraj.distance(traj, commit_define_fwd[0][i] + ' ' + commit_define_fwd[1][i], n_frames=1)[-1] >= commit_define_fwd[2][i]:
                commit_flag = 'fwd'
            else:
                commit_flag = ''
                break
        else:
            open('as.log', 'a').write('\nAn incorrect commitor definition \"' + commit_define_fwd[3][i] + '\" was given for index ' + str(i) +' in the \'fwd\' direction.')
            sys.exit('An incorrect commitor definition \"' + commit_define_fwd[3][i] + '\" was given for index ' + str(i) +' in the \'fwd\' direction.')

    if commit_flag == '':                # only bother checking for bwd commitment if not fwd commited
        for i in range(0,len(commit_define_bwd[2])):
            if commit_define_bwd[3][i] == 'lt':
                if pytraj.distance(traj, commit_define_bwd[0][i] + ' ' + commit_define_bwd[1][i], n_frames=1)[-1] <= commit_define_bwd[2][i]:
                    commit_flag = 'bwd'  # if a commitor test is passed, testing moves on to the next one.
                else:
                    commit_flag = ''
                    break                # if a commitor test is not passed, all testing in this direction ends in a fail
            elif commit_define_bwd[3][i] == 'gt':
                if pytraj.distance(traj, commit_define_bwd[0][i] + ' ' + commit_define_bwd[1][i], n_frames=1)[-1] >= commit_define_bwd[2][i]:
                    commit_flag = 'bwd'
                else:
                    commit_flag = ''
                    break
            else:
                open('as.log', 'a').write('\nAn incorrect commitor definition \"' + commit_define_bwd[3][i] + '\" was given for index ' + str(i) +' in the \'bwd\' direction.')
                sys.exit('An incorrect commitor definition \"' + commit_define_bwd[3][i] + '\" was given for index ' + str(i) +' in the \'bwd\' direction.')

    return commit_flag


def pickframe(thread,direction):
    # Picks a random frame from the trajectory given by the thread in the given direction (either 'fwd' or 'bwd'). If
    # which is 'this', the trajectory chosen from is the most recent; if it's 'last', then it's the one before. The
    # selection is chosen from within the first n_adjust steps. Once the desired frame is chosen, pytraj is used to
    # extract it from the given trajectory and the name of the new shooting point .rst file is returned sans extension.
    if thread.last_valid == '0':    # to catch case where 'last_valid' is still the initial shooting point because a new valid transition path has not been found
        return thread.start_name    # will cause the new start_name to be unchanged from the previous run

    frame_number = random.randint(0,n_adjust)
    this_suffix = thread.last_valid
    traj = pytraj.iterload(thread.basename + '_' + this_suffix + '_' + direction + '.nc', thread.prmtop, format='.nc', frame_slice=[(frame_number, frame_number)])
    new_suffix = str(int(thread.suffix) + 1)
    pytraj.write_traj(thread.basename + '_' + new_suffix + '.rst',traj,format='.rst7')

    open('as.log', 'a').write('\nInitializing next shooting point from shooting run ' + thread.basename + '_' + str(thread.last_valid) + ' in ' + direction + ' direction, frame number ' + str(frame_number))

    return thread.basename + '_' + new_suffix


def cleanthread(thread):
    # Function to modify/clean-up thread attributes in preparation for the next iteration of the thread, and then add it
    # to the itinerary. Also cancels the current jobs in this thread, since they have already committed.
    direction = random.randint(0,1)     # I'm not sure doing this helps, but I am sure doing it doesn't hurt
    if direction == 0:
        pick_dir = 'fwd'
    else:
        pick_dir = 'bwd'

    # Record result of forward trajectory in output file. This is done regardless of whether the shooting point was
    # accepted; accept/reject is for keeping the sampling around the separatrix, but even rejected points are valid for
    # calculating the reaction coordinate!
    if thread.commit1 != 'fail':
        if thread.commit1 == 'fwd':
            basin = 'A'
        elif thread.commit1 == 'bwd':
            basin = 'B'
        open('as.out', 'a').write(basin + ' <- ' + candidatevalues(thread) + '\n')

    thread.total_moves += 1
    open('as.log', 'a').write('\nShooting run ' + thread.name + ' finished with fwd trajectory result: ' + thread.commit1 + ' and bwd trajectory result: ' + thread.commit2)
    open('as.log', 'a').write('\n' + thread.name + ' has a current acceptance ratio of: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or ' + str(100*thread.accept_moves/thread.total_moves) + '%')

    # Cancel current jobs in thread; THIS SHOULD NOT WORK, because the jobid values have already been wiped when this is called
    # process1 = subprocess.Popen(['scancel', thread.jobid1], stdout=subprocess.PIPE)
    # (output1, err1) = process1.communicate()
    # process2 = subprocess.Popen(['scancel', thread.jobid2], stdout=subprocess.PIPE)
    # (output2, err2) = process2.communicate()

    thread.start_name = pickframe(thread, pick_dir)
    thread.type = 'init'
    thread.suffix = str(int(thread.suffix) + 1)
    thread.name = thread.basename + '_' + thread.suffix
    thread.jobid1 = ''  # should be redundant, but no reason not to double-up just in case.
    thread.jobid2 = ''  # should be redundant, but no reason not to double-up just in case.
    thread.commit1 = ''
    thread.commit2 = ''
    itinerary.append(thread)


def candidatevalues(thread):
    # Returns a space-separated string containing all the candidate OP values requested by the user in the form of the
    # candidateops list object.
    output = ''
    traj = pytraj.iterload(thread.start_name, thread.prmtop)
    for index in range(0,len(candidateops[0])):
        value = pytraj.distance(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index])
        output += str(value) + ' '

    return output


def revvels(thread):
    # Reads the fwd restart file corresponding to a given thread and produces a duplicate restart file with all the
    # velocities multiplied by -1. Velocities in .rst files are stored in groups of three just like coordinates and
    # directly following them. This function will find the place where coordinates end and velocities begin by reading
    # the number of atoms listed on the 2nd line of the .rst file, dividing it by two, and then navigating to the line
    # immediately after that plus three (since coordinates begin on line 3) and minus one (because of indexing)
    byline = open(thread.name + '_init_fwd.rst').readlines()
    pattern = re.compile('[-0-9.]+')        # regex to match numbers including decimals and negatives
    pattern2 = re.compile('\s[-0-9.]+')     # regex to match numbers including decimals and negatives, with one space in front
    n_atoms = pattern.findall(byline[1])[0] # number of atoms indicated on second line of .rst file

    shutil.copyfile(thread.name + '_init_fwd.rst', thread.name + '_init_bwd.rst')
    for i, line in enumerate(fileinput.input(thread.name + '_init_bwd.rst', inplace=1)):
        if i >= int(n_atoms)/2 + 2 and i <= int(n_atoms)*2:     # if this line is a velocity line
            newline = line
            for vel in pattern2.findall(newline):
                if '-' in vel:
                    newline.replace(vel, '  ' + vel[2:], 1)     # replace ' -magnitude' with '  magnitude'
                else:
                    newline.replace(vel, '-' + vel[1:], 1)      # replace ' magnitude' with '-magnitude'
            sys.stdout.write(newline)
        else:                                                   # if not a velocity line
            sys.stdout.write(line)                              # return unmodified line



# Then I want to define a loop that defines the runtime of the program. This loop will...
#   assemble a list of all the jobs it needs to run,
#   make the corresponding batch files and submit them, collecting the jobID's into a list,
#   intermittently check for completion of jobs in its list, add the next step to the to-do list, and restart,
#   before eventually terminating when its to-do list is empty and it has no outstanding jobs.

itinerary = []  # a list of threads that need running
running = []    # a list of currently running threads

for structure in start_name:  # for all of the initial structures...
    thread = spawnthread(structure,suffix='1')      # spawn a new thread with the default settings
    thread.last_valid = '0'                         # so that if the first shooting point does not result in a valid transition path, shooting will begin from the TS guess
    thread.prmtop = topology                        # set prmtop filename for the thread
    itinerary.append(thread)                        # submit it to the itinerary
    shutil.copy('../' + structure, './')            # and copy the input structure to the working directory...
    shutil.copy('../' + thread.prmtop, './')        # ... and its little topology file, too!

try:
    os.remove('as.log')                 # delete previous run's log (this should no longer be necessary, as the whole working directory gets deleted at the start of each run)
except OSError:                         # catches error if no previous log file exists
    pass
with open('as.log', 'w+') as newlog:
    newlog.write('New log file')        # make new log file
    newlog.close()

try:
    os.remove('as.out')                 # delete previous run's output file (this should no longer be necessary, as the whole working directory gets deleted at the start of each run)
except OSError:                         # catches error if no previous output file exists
    pass
with open('as.out', 'w+') as newout:    # make a new output file
    newout.close()


if batch_system == 'pbs':
    user_alias = '$USER'
    queue_command = 'qselect -u ' + user_alias + ' -s QR'
    cancel_command = 'qdel'
elif batch_system == 'slurm':
    user_alias = '$USER'
    queue_command = 'squeue -u ' + user_alias
    cancel_command = 'scancel'

while itinerary or running:  # while either list has contents...
    itin_names = [thread.name + '_' + thread.type for thread in itinerary]
    run_names = [thread.name + '_' + thread.type for thread in running]
    open('as.log', 'a').write('\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))
    open('as.log', 'a').write('\nSubmitting jobs in itinerary...')

    index = -1                                  # set place in itinerary to -1
    for thread in itinerary:                    # for each thread that's ready for its next step...
        index += 1                              # increment place in itinerary
        makebatch(thread)                       # make the necessary batch file
        if thread.type == 'init':
            thread.jobid1 = subbatch(thread)    # submit that batch file and collect its jobID into thread
        else:
            thread.jobid1 = subbatch(thread, 'fwd')
            thread.jobid2 = subbatch(thread, 'bwd')
        running.append(itinerary[index])        # move element of itinerary into next position in running

    itinerary = []  # empty itinerary

    itin_names = [thread.name + '_' + thread.type for thread in itinerary]
    run_names = [thread.name + '_' + thread.type for thread in running]
    open('as.log', 'a').write('\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))

    while not itinerary:                    # while itinerary is empty...
        process = subprocess.Popen(queue_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, close_fds=True,
                                   shell=True)  # check list of currently running jobs
        # The above line retrieves both the stdout and stderr streams into the same variable; on PBS, sometimes this
        # returns a "busy" message. The following while loop is meant to handle that, but it's obviously ugly.
        output = process.stdout.read()
        while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):  # todo: fix this kludge
            process = subprocess.Popen(queue_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, close_fds=True, shell=True)
            output = process.stdout.read()

        index = -1                          # set place in running to -1
        for thread in running:
            index += 1                      # increment place in running
            if not thread.jobid1 in str(output): # if one of the submitted jobs no longer appears to be running, todo: add functionality to catch jobs that exit with non-zero exit code and handle them appropriately
                if thread.type == 'init':
                    try:
                        revvels(thread)         # make the initial .rst file for the bwd trajectory
                        itinerary.append(running[index])
                        thread.type = 'prod'
                        open('as.log', 'a').write('\nJob completed: ' + thread.name + '_init\nAdding ' + thread.name + ' forward and backward jobs to itinerary')
                        del running[index]
                        index -= 1              # to keep index on track after deleting an entry
                    except FileNotFoundError:
                        open('as.log', 'a').write('\nThread ' + thread.basename + ' crashed: initialization did not produce a restart file.')
                        if restart_on_crash == False:
                            open('as.log', 'a').write('\nrestart_on_crash == False; thread will not restart')
                        elif restart_on_crash == True:
                            open('as.log', 'a').write('\nrestart_on_crash == True; resubmitting thread to itinerary')
                            itinerary.append(running[index])
                            thread.type = 'init'
                        del running[index]
                        index -= 1  # to keep index on track after deleting an entry
                elif thread.type == 'prod':
                    # fwd trajectory exited before passing a commitor test, either walltime or other error
                    thread.commit1 = checkcommit(thread, 'fwd') # check one last time
                    if not thread.commit1:
                        thread.commit1 = 'fail'
                    thread.jobid1 = ''
            if not thread.jobid2 in str(output): # if one of the submitted jobs no longer appears to be running, todo: add functionality to catch jobs that exit with non-zero exit code and handle them appropriately
                # bwd trajectory exited before passing a commitor test, either walltime or other error
                thread.commit2 = checkcommit(thread, 'bwd')  # check one last time
                if not thread.commit2:
                    thread.commit2 = 'fail'
                thread.jobid2 = ''

        flag = False
        index = -1
        for thread in running:
            index += 1
            if thread.type == 'prod':
                if not thread.commit1:
                    thread.commit1 = checkcommit(thread,'fwd')
                if not thread.commit2:
                    thread.commit2 = checkcommit(thread,'bwd')
                if thread.commit1 and thread.jobid1:
                    process = subprocess.Popen([cancel_command, thread.jobid1], stdout=subprocess.PIPE)
                    (output, err) = process.communicate()   # doesn't do anything, I think
                    thread.jobid1 = ''
                if thread.commit2 and thread.jobid2:
                    process = subprocess.Popen([cancel_command, thread.jobid2], stdout=subprocess.PIPE)
                    (output, err) = process.communicate()   # doesn't do anything, I think
                    thread.jobid2 = ''
                if thread.commit1 and thread.commit2:
                    flag = True
                    del running[index]
                    index -= 1                              # to keep index on track after deleting an entry
                    if thread.commit1 != thread.commit2 and thread.commit1 != 'fail' and thread.commit2 != 'fail': # valid transition path, update 'last_valid' attribute
                        thread.last_valid = thread.suffix
                        thread.accept_moves += 1
                    cleanthread(thread)

                    # todo: add helpful log entry here
        if not flag:
            time.sleep(60)  # Delay 60 seconds before checking for job status again

    if not itinerary and not running:
        open('as.log', 'a').write('\nItinerary and running lists are empty\nAimless shooting is complete! See as.out in the working directory for results.')
        break
