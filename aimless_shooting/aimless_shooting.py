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
import importlib
import pickle
from jinja2 import Environment, FileSystemLoader
rc_eval = importlib.import_module('rc_eval')
import tracemalloc

tracemalloc.start()


# Define the "thread" object that constitutes one string of shooting moves in our search through phase space. Each
# thread is an independent series of simulations with its own acceptance ratio, and may or may not have a unique
# starting structure.
class Thread:
    def __init__(self):
        self.basename = ''      # first part of name, universal to all files in the thread
        self.name = ''          # full name, identical to basename unless otherwise specified
        self.suffix = ''        # appended to basename after an underscore to build name. Should be str(an integer)
        self.jobid1 = ''        # current batch system jobid associated with the thread init step or forward trajectory
        self.jobid2 = ''        # another jobid slot, for the backward trajectory
        self.type = ''          # thread job type; init to get initial trajectories or prod to run forward and backward
        self.start_name = ''    # name of previous thread to initialize shooting from
        self.last_valid = ''    # suffix of the most recent shooting point that resulted in a valid transition path
        self.commit1 = ''       # flag indicating commitment direction for the fwd trajectory
        self.commit2 = ''       # flag indicating commitment direction for the bwd trajectory
        self.accept_moves = 0   # running total of completed shooting moves that were accepted
        self.total_moves = 0    # running total of completed shooting moves with any result
        self.prmtop = ''        # name of parameter/topology file corresponding to this thread
        self.failcount = 0      # number of unaccepted shooting moves in a row for this thread
        self.status = ''        # string to indicate which termination criterion ended the thread, if applicable
        self.history = []       # for each job in this threads history, includes job name and result code: F = both went forward, B = both went backward, S = success, and X = one or both simulations failed
        self.commitlist = []    # for use in committor analysis
        self.jobidlist = []     # for use in committor analysis


# Then, define a series of functions to perform the basic tasks of creating batch files, interacting with the batch
# system, and interpreting/modifying coordinate and trajectory files.
def makebatch(thread,n_shots=0):
    # Makes batch files for the three jobs that constitute one shooting move, using Jinja2 template filling
    # n_shots and type == committor_analysis are used in rc_eval.py

    name = thread.name
    type = thread.type
    batch = 'batch_' + batch_system + '.tpl'

    if not os.path.exists(home_folder + '/' + 'input_files'):
        sys.exit('Error: could not locate input_files folder: ' + home_folder + '/' + 'input_files\nSee documentation '
                                                                                      'for the \'home_folder\' option.')

    if type == 'init':
        # init batch file
        open('as.log', 'a').write('\nWriting init batch file for ' + name + ' starting from ' + thread.start_name)
        template = env.get_template(batch)
        filled = template.render(name=name + '_init', nodes=init_nodes, taskspernode=init_ppn, walltime=init_walltime,
                                 solver='sander', inp=home_folder + '/input_files/init.in', out=name + '_init.out',
                                 prmtop=thread.prmtop, inpcrd=thread.start_name, rst=name + '_init_fwd.rst',
                                 nc=name + '_init.nc', mem=init_mem, working_directory=working_directory)
        with open(name + '_init.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

    elif type == 'prod':
        # forward and backward simulation batch files
        open('as.log', 'a').write('\nWriting forward batch file for ' + name)
        template = env.get_template(batch)
        filled = template.render(name=name + '_fwd', nodes=prod_nodes, taskspernode=prod_ppn, walltime=prod_walltime,
                                 solver='sander', inp=home_folder + '/input_files/prod.in', out=name + '_fwd.out',
                                 prmtop=thread.prmtop, inpcrd=name + '_init_fwd.rst', rst=name + '_fwd.rst',
                                 nc=name + '_fwd.nc', mem=prod_mem, working_directory=working_directory)
        with open(name + '_fwd.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

        open('as.log', 'a').write('\nWriting backward batch file for ' + name)
        template = env.get_template(batch)
        filled = template.render(name=name + '_bwd', nodes=prod_nodes, taskspernode=prod_ppn, walltime=prod_walltime,
                                 solver='sander', inp=home_folder + '/input_files/prod.in', out=name + '_bwd.out',
                                 prmtop=thread.prmtop, inpcrd=name + '_init_bwd.rst', rst=name + '_bwd.rst',
                                 nc=name + '_bwd.nc', mem=prod_mem, working_directory=working_directory)
        with open(name + '_bwd.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

    elif type == 'committor_analysis':
        name = thread.basename
        for i in range(n_shots):
            template = env.get_template(batch)
            filled = template.render(name=name + '_ca_' + str(i), nodes=prod_nodes, taskspernode=prod_ppn,
                                     walltime=prod_walltime, solver='sander', inp=home_folder + '/input_files/prod.in',
                                     out=name + '_ca_' + str(i) + '.out', prmtop=thread.prmtop, inpcrd=name,
                                     rst=name + '_ca_' + str(i) + '.rst', nc=name + '_ca_' + str(i) + '.nc',
                                     mem=prod_mem, working_directory=working_directory)
            with open(name + '_ca_' + str(i) + '.' + batch_system, 'w') as newfile:
                newfile.write(filled)
                newfile.close()

    else:
        open('as.log', 'a').write('\nAn incorrect job type \"' + type + '\" was passed to makebatch.')
        sys.exit('An incorrect job type \"' + type + '\" was passed to makebatch.')


def subbatch(thread,direction = '',n_shots=0):
    # Submits a batch file given by batch and returns its process number as a string
    # n_shots and type == committor_analysis are used in rc_eval.py
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
        open('as.log', 'a').write('\nAn incorrect batch system type \"' + batch_system + '\" was passed to subbatch.')
        sys.exit('An incorrect batch system type \"' + batch_system + '\" was supplied. Acceptable types are: pbs, slurm')

    if type == 'committor_analysis':
        jobids = []
        for i in range(n_shots):
            if batch_system == 'pbs':
                command = 'qsub ' + name + '_ca_' + str(i) + '.' + batch_system
            elif batch_system == 'slurm':
                command = 'sbatch ' + name + '_ca_' + str(i) + '.' + batch_system
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       close_fds=True, shell=True)
            output = process.stdout.read()
            while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           close_fds=True, shell=True)
                output = process.stdout.read()
            if 'Bad UID for job execution MSG=user does not exist in server password file' in str(output):
                open('as.log', 'a').write('\nWarning: attempted to submit a job, but got error: ' + str(output) + '\n'
                                          + 'Retrying in one minute...')
                time.sleep(60)
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           close_fds=True, shell=True)
                output = process.stdout.read()
            pattern = re.compile('[0-9]+')
            try:
                jobids.append(pattern.findall(str(output))[0])
            except IndexError:
                sys.exit('Error: unable to submit a batch job. Got message: ' + str(output))
        return jobids
    else:
        open('as.log', 'a').write('\nSubmitting job: ' + name + '_' + type + '.' + batch_system)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read()
        while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       close_fds=True, shell=True)
            output = process.stdout.read()
        open('as.log', 'a').write('\nBatch system says: ' + str(output))
        pattern = re.compile('[0-9]+')
        try:
            return pattern.findall(str(output))[0]
        except IndexError:
            sys.exit('Error: unable to submit a batch job. Got message: ' + str(output))


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

    if direction not in ['fwd','bwd']:  # occurs when this is called by rc_eval.py
        name = thread.basename
    else:
        name = thread.name

    if not os.path.isfile(name + '_' + direction + '.nc'):   # if the file doesn't exist yet, just do nothing
        return ''

    traj = pytraj.iterload(name + '_' + direction + '.nc', thread.prmtop, format='.nc')

    if not traj:                        # catches error if the trajectory file exists but has zero frames
        print('Don\'t worry about this internal error; it just means that aimless_shooting is checking for commitment in a trajectory that doesn\'t have any frames yet, probably because the simulation has only just begun.')
        return ''

    commit_flag = ''                    # initialize flag for commitment; this is the value to be returned eventually
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

    del traj    # to ensure cleanup of iterload object

    return commit_flag


def pickframe(thread,direction,forked_from=Thread()):
    # Picks a random frame from the last valid trajectory of thread in the given direction (either 'fwd' or 'bwd'). The
    # selection is chosen from within the first n_adjust steps. Once the desired frame is chosen, pytraj is used to
    # extract it from the given trajectory and the name of the new shooting point .rst file is returned sans extension.
    if thread.last_valid == '0':    # to catch case where 'last_valid' is still the initial shooting point because a new valid transition path has not been found
        return thread.start_name    # will cause the new start_name to be unchanged from the previous run

    frame_number = random.randint(1, n_adjust)

    if forked_from.basename:    # if an actual thread was given for forked_from (as opposed to the default)...
        traj = pytraj.iterload(forked_from.basename + '_' + forked_from.last_valid + '_' + direction + '.nc', forked_from.prmtop, format='.nc', frame_slice=[(frame_number, frame_number + 1)])
        new_suffix = '1'
    else:
        traj = pytraj.iterload(thread.basename + '_' + thread.last_valid + '_' + direction + '.nc', thread.prmtop, format='.nc', frame_slice=[(frame_number, frame_number + 1)])
        new_suffix = str(int(thread.suffix) + 1)

    new_restart_name = thread.basename + '_' + new_suffix + '.rst7'
    pytraj.write_traj(new_restart_name,traj,options='multi')    # multi because not including this option seems to keep it on anyway, so I want to be consistent
    try:
        os.rename(new_restart_name + '.1',new_restart_name)     # I don't quite know why, but pytraj appends '.1' to the filename, so this removes it.
    except OSError: # I sort of anticipate this breaking down the line, so this block is here to help handle that.
        open('as.log', 'a').write('\nWarning: tried renaming .rst7.1 file to .rst7, but encountered OSError exception. Either you ran out of storage space, or this is a possible indication of an unexpected pytraj version?')
        if not os.path.exists(new_restart_name):
            sys.exit('\nError: pickframe did not produce the restart file for the next shooting move. Please ensure that you didn\'t run out of storage space, and then raise this issue on GitHub to let me know!')
        else:
            open('as.log', 'a').write('\nWarning: it tentatively looks like this should be okay, as the desired file was still created.')
        pass

    if forked_from.basename:
        open('as.log', 'a').write('\nForking ' + thread.basename + ' from ' + forked_from.basename + '_' + forked_from.last_valid + '_' + direction + '.nc, frame number ' + str(frame_number))
    else:
        open('as.log', 'a').write('\nInitializing next shooting point from shooting run ' + thread.basename + '_' + thread.last_valid + ' in ' + direction + ' direction, frame number ' + str(frame_number))

    del traj  # to ensure cleanup of iterload object

    return new_restart_name


def cleanthread(thread):
    # Function to modify/clean-up thread attributes in preparation for the next iteration of the thread, and then add it
    # to the itinerary. Also cancels the current jobs in this thread, since they have already committed.

    # Record result of forward trajectory in output file. This is done regardless of whether the shooting point was
    # accepted; accept/reject is for keeping the sampling around the separatrix, but even rejected points are valid for
    # calculating the reaction coordinate so long as they committed to a basin!
    if thread.commit1 != 'fail':
        if thread.commit1 == 'fwd':
            basin = 'A'
        elif thread.commit1 == 'bwd':
            basin = 'B'
        else:
            basin = thread.commit1
            sys.exit('Error: thread commit1 flag took on unexpected value: ' + basin + '\nThis is a weird error. Please raise this issue on GitHub!')
        open('as.out', 'a').write(basin + ' <- ' + candidatevalues(thread) + '\n')

    # Write last result to history
    if thread.last_valid == thread.suffix:
        code = 'S'
    elif thread.commit1 == thread.commit2 == 'fwd':
        code = 'F'
    elif thread.commit1 == thread.commit2 == 'bwd':
        code = 'B'
    else:
        code = 'X'
    thread.history.append(thread.name + ' ' + code)
    with open('history/' + thread.basename, 'w') as file:
        for history_line in thread.history:
            file.write(history_line + '\n')

    thread.total_moves += 1
    open('as.log', 'a').write('\nShooting run ' + thread.name + ' finished with fwd trajectory result: ' + thread.commit1 + ' and bwd trajectory result: ' + thread.commit2)
    open('as.log', 'a').write('\n' + thread.basename + ' has a current acceptance ratio of: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or ' + str(100*thread.accept_moves/thread.total_moves)[0:5] + '%')

    # Implementation of fork. Makes (fork - 1) new threads from successful runs and adds them to the itinerary. The new
    # threads do not inherit anything from their parents except starting point and history.
    if fork > 1 and thread.last_valid == thread.suffix:
        for i in range(fork - 1):
            direction = random.randint(0, 1)
            if direction == 0:
                pick_dir = 'fwd'
            else:
                pick_dir = 'bwd'
            newthread = spawnthread(thread.name + '_' + str(i + 1),suffix='1')
            newthread.prmtop = thread.prmtop
            newthread.start_name = pickframe(newthread, pick_dir, thread)
            newthread.last_valid = '0'
            newthread.history = thread.history
            itinerary.append(newthread)
            allthreads.append(newthread)

    direction = random.randint(0, 1)  # I'm not sure doing this helps, but I am sure doing it doesn't hurt
    if direction == 0:
        pick_dir = 'fwd'
    else:
        pick_dir = 'bwd'

    if thread.last_valid == thread.suffix or always_new:  # pick a new starting point iff the last move was a success
        thread.start_name = pickframe(thread, pick_dir)
    thread.type = 'init'
    thread.suffix = str(int(thread.suffix) + 1)
    thread.name = thread.basename + '_' + thread.suffix
    thread.jobid1 = ''  # should be redundant, but no reason not to double-up just in case.
    thread.jobid2 = ''  # should be redundant, but no reason not to double-up just in case.
    thread.commit1 = ''
    thread.commit2 = ''

    if thread.failcount >= max_fails > 0:
        thread.status = 'max_fails'     # the thread dies because it has failed too many times in a row
    elif thread.total_moves >= max_moves > 0:
        thread.status = 'max_moves'     # the thread dies because it has performed too many total moves
    elif thread.accept_moves >= max_accept > 0:
        thread.status = 'max_accept'    # the thread dies because it has accepted too many moves
    else:
        global itinerary
        itinerary.append(thread)        # the thread lives and moves to next step

    # Write a status file to indicate the acceptance ratio and current status of every thread.
    with open('status.txt','w') as file:
        for thread in allthreads:
            try:
                file.write(thread.basename + ' acceptance ratio: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or ' + str(100*thread.accept_moves/thread.total_moves)[0:5] + '%\n')
            except ZeroDivisionError:   # Since any thread that hasn't completed a move yet has total_moves = 0
                file.write(thread.basename + ' acceptance ratio: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or 0%\n')
            if thread in itinerary:
                file.write('  Status: move ' + thread.suffix + ' queued\n')
            elif thread in running:
                file.write('  Status: move ' + thread.suffix + ' running\n')
            else:
                if thread.status in ['max_accept','max_moves','max_fails']:
                    file.write('  Status: terminated after move ' + thread.suffix + ' due to termination criterion: ' + thread.status + '\n')
                else:
                    file.write('  Status: crashed during move ' + thread.suffix + '\n')


def candidatevalues(thread):
    # Returns a space-separated string containing all the candidate OP values requested by the user in the form of the
    # candidateops list object.
    output = ''
    traj = pytraj.iterload(thread.start_name, thread.prmtop)
    for index in range(0,len(candidateops[0])):
        if len(candidateops) == 4:          # candidateops contains dihedrals
            if candidateops[3][index]:      # if this OP is a dihedral
                value = pytraj.dihedral(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' + candidateops[2][index] + ' ' + candidateops[3][index])
            elif candidateops[2][index]:    # if this OP is an angle
                value = pytraj.angle(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' + candidateops[2][index])
            else:                           # if this OP is a distance
                value = pytraj.distance(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index])
            output += str(value) + ' '
        elif len(candidateops) == 3:        # candidateops contains angles but not dihedrals
            if candidateops[2][index]:      # if this OP is an angle
                value = pytraj.angle(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' + candidateops[2][index])
            else:                           # if this OP is a distance
                value = pytraj.distance(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index])
            output += str(value) + ' '
        else:                               # candidateops contains only distances
            value = pytraj.distance(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index])
            output += str(value) + ' '

    del traj  # to ensure cleanup of iterload object

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
    offset = 2                              # appropriate for n_atoms is odd; offset helps avoid modifying the box line
    if int(n_atoms) % 2 == 0:               # if n_atoms is even...
        offset = 1                          # appropriate for n_atoms is even

    shutil.copyfile(thread.name + '_init_fwd.rst', thread.name + '_init_bwd.rst')
    for i, line in enumerate(fileinput.input(thread.name + '_init_bwd.rst', inplace=1)):
        if i >= int(n_atoms)/2 + 2 and i <= int(n_atoms) + offset:      # if this line is a velocity line
            newline = line
            for vel in pattern2.findall(newline):
                if '-' in vel:
                    newline = newline.replace(vel, '  ' + vel[2:], 1)   # replace ' -magnitude' with '  magnitude'
                else:
                    newline = newline.replace(vel, '-' + vel[1:], 1)    # replace ' magnitude' with '-magnitude'
            sys.stdout.write(newline)
        else:                                                           # if not a velocity line
            sys.stdout.write(line)

# Then, define a loop that constitutes the runtime of the program. This loop will...
#   assemble a list of all the jobs it needs to run,
#   make the corresponding batch files and submit them, collecting the jobID's into a list,
#   intermittently check for completion of jobs in its list, add the next step to the to-do list, and restart,
#   before eventually terminating when its to-do list is empty and it has no outstanding jobs.


def main_loop():
    # This is the primary runtime loop of the code. It is encapsulated inside this function so that it can be called
    # directly when restart == True rather than going through all the motions of initializing the workspace beforehand.

    # Commands to issue to the shell to check for job status and to cancel completed jobs.
    # Note that a copy of this block appears in rc_eval.py, so editing one does not affect the other.
    user_alias = '$USER'
    if batch_system == 'pbs':
        queue_command = 'qselect -u ' + user_alias + ' -s QR'
        cancel_command = 'qdel'
    elif batch_system == 'slurm':
        queue_command = 'squeue -u ' + user_alias
        cancel_command = 'scancel'

    global itinerary
    global running
    global allthreads

    while itinerary or running:  # while either list has contents...
        itin_names = [thread.name + '_' + thread.type for thread in itinerary]
        run_names = [thread.name + '_' + thread.type for thread in running]
        open('as.log', 'a').write(
            '\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))
        open('as.log', 'a').write('\nSubmitting jobs in itinerary...')

        index = -1  # set place in itinerary to -1
        for thread in itinerary:  # for each thread that's ready for its next step...
            index += 1  # increment place in itinerary
            makebatch(thread)  # make the necessary batch file
            if thread.type == 'init':
                thread.jobid1 = subbatch(thread)  # submit that batch file and collect its jobID into thread
            else:
                thread.jobid1 = subbatch(thread, 'fwd')
                thread.jobid2 = subbatch(thread, 'bwd')
            running.append(itinerary[index])  # move element of itinerary into next position in running

        itinerary = []  # empty itinerary

        itin_names = [thread.name + '_' + thread.type for thread in itinerary]
        run_names = [thread.name + '_' + thread.type for thread in running]
        open('as.log', 'a').write('\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))

        while not itinerary:  # while itinerary is empty...
            process = subprocess.Popen(queue_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, close_fds=True,
                                       shell=True)  # check list of currently running jobs
            # The above line retrieves both the stdout and stderr streams into the same variable; on PBS, sometimes this
            # returns a "busy" message. The following while loop is meant to handle that, but it's obviously ugly.
            output = process.stdout.read()
            while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
                process = subprocess.Popen(queue_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, close_fds=True, shell=True)
                output = process.stdout.read()

            index = 0  # set place in running to 0
            while index < len(running):  # while instead of for to control indexing manually
                thread = running[index]
                if thread.jobid1 and not thread.jobid1 in str(output):  # if a submitted job is no longer running
                    if thread.type == 'init':
                        try:
                            revvels(thread)  # make the initial .rst file for the bwd trajectory
                            itinerary.append(running[index])
                            thread.type = 'prod'
                            open('as.log', 'a').write(
                                '\nJob completed: ' + thread.name + '_init\nAdding ' + thread.name + ' forward and backward jobs to itinerary')
                            del running[index]
                            index -= 1  # to keep index on track after deleting an entry
                        except FileNotFoundError:  # when revvels can't find the init .rst file
                            open('as.log', 'a').write('\nThread ' + thread.basename + ' crashed: initialization did not produce a restart file.')
                            if restart_on_crash == False:
                                open('as.log', 'a').write('\nrestart_on_crash = False; thread will not restart')
                            elif restart_on_crash == True:
                                open('as.log', 'a').write('\nrestart_on_crash = True; resubmitting thread to itinerary')
                                itinerary.append(running[index])
                                thread.type = 'init'
                            del running[index]
                            index -= 1  # to keep index on track after deleting an entry
                    elif thread.type == 'prod':
                        # fwd trajectory exited before passing a commitor test, either walltime or other error
                        thread.commit1 = checkcommit(thread, 'fwd')  # check one last time
                        if not thread.commit1:
                            thread.commit1 = 'fail'
                        thread.jobid1 = ''
                if thread.jobid2 and not thread.jobid2 in str(output):  # if one of the submitted jobs no longer appears to be running
                    # bwd trajectory exited before passing a commitor test, either walltime or other error
                    thread.commit2 = checkcommit(thread, 'bwd')  # check one last time
                    if not thread.commit2:
                        thread.commit2 = 'fail'
                    thread.jobid2 = ''
                index += 1  # increment place in running

            index = 0
            while index < len(running):
                thread = running[index]
                if thread.type == 'prod':
                    if not thread.commit1:
                        thread.commit1 = checkcommit(thread, 'fwd')
                    if not thread.commit2:
                        thread.commit2 = checkcommit(thread, 'bwd')
                    if thread.commit1 and thread.jobid1:
                        process = subprocess.Popen([cancel_command, thread.jobid1], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()  # doesn't do anything, I think
                        thread.jobid1 = ''
                    if thread.commit2 and thread.jobid2:
                        process = subprocess.Popen([cancel_command, thread.jobid2], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()  # doesn't do anything, I think
                        thread.jobid2 = ''
                    if thread.commit1 and thread.commit2:
                        del running[index]
                        index -= 1  # to keep index on track after deleting an entry
                        thread.failcount += 1  # increment fails in a row regardless of outcome
                        if thread.commit1 != thread.commit2 and thread.commit1 != 'fail' and thread.commit2 != 'fail':  # valid transition path, update 'last_valid' attribute
                            thread.last_valid = thread.suffix
                            thread.accept_moves += 1
                            thread.failcount = 0  # reset fail count to zero if this move was accepted
                        cleanthread(thread)
                index += 1

            # For tracemalloc memory leak debugging
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

            pickle.dump(allthreads, open('restart.pkl', 'wb'))
            time.sleep(60)  # Delay 60 seconds before checking for job status again

        if not itinerary and not running:
            acceptances = [(100 * thread.accept_moves / thread.total_moves) for thread in allthreads]
            max_of_accepts = max(acceptances)
            open('as.log', 'a').write('\nItinerary and running lists are empty.\nAimless shooting is complete! The highest acceptance ratio for any thread was ' + max_of_accepts + '%.\nSee as.out in the working directory for results.')
            break


# Get directory from which the code was called
called_path = os.getcwd()

# Parse arguments from command line using argparse
parser = argparse.ArgumentParser(description='Perform aimless shooting according to the settings given in the input file.')
parser.add_argument('-O', action='store_true', help='flag indicating that existing working_directory should be overwritten if it exists.')
parser.add_argument('-i', metavar='input_file', type=str, nargs=1, default='as.in', help='input filename; see documentation for format. Default=\'as.in\'')
parser.add_argument('-w', metavar='working_directory', type=str, nargs=1, default=os.getcwd() + '/as_working', help='working directory. Default=\'`pwd`/as_working\'')
arguments = vars(parser.parse_args())   # Retrieves arguments as a dictionary object

overwrite = arguments.get('O')

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
working_directory = arguments.get('w')          # Working directory for aimless shooting calculations
restart_on_crash = False                        # If a thread crashes during initialization, should it be resubmitted?
max_fails = 10                                  # Number of unaccepted shooting moves before a thread is killed; a negative value means "no max"
max_moves = 100                                 # Number of moves with any result permitted before the thread is terminated
max_accept = 100                                # Number of accepted moves permitted before the thread is terminated
degeneracy = 1                                  # Number of duplicate threads to produce for each initial structure
init_nodes = 1                                  # Number of nodes on which to run initialization jobs
init_ppn = 1                                    # Number of processors per node on which to run initialization jobs
init_walltime = '01:00:00'                      # Wall time for initialization jobs
init_mem = '4000mb'                             # Memory per core for intialization jobs
prod_nodes = 1                                  # Number of nodes on which to run production jobs
prod_ppn = 1                                    # Number of processors per node on which to run production jobs
prod_walltime = '01:00:00'                      # Wall time for production jobs
prod_mem = '4000mb'                             # Memory per core for intialization jobs
resample = False                                # If True, don't run any new simulations; just rewrite as.out based on current settings and existing data
fork = 1                                        # Number of new threads to spawn after each successful shooting move. Each one gets its own call to pickframe()
home_folder = os.path.dirname(os.path.realpath(sys.argv[0]))    # Directory containing templates and input_files folders
always_new = False                              # Pick a new shooting move after every move, even if it isn't accepted?
rc_definition = ''                              # Defines the equation of the reaction coordinate for an rc_eval run
rc_minmax = ''                                  # Minimum and maximum values of each OP, for use in rc_eval variable reduction
candidateops = ''                               # Defines the candidate order parameters to output into the as.out file
committor_analysis = ''                         # Variables to pass into committor analysis.
restart = False                                 # Whether or not to restart an old AS run located in working_directory

if type(working_directory) == list:             # handles inconsistency in format when the default value is used vs. when a value is given
    working_directory = working_directory[0]


def str2bool(var):                              # Function to convert string "True" or "False" to corresponding boolean
    return str(var).lower() in ['true']         # returns False for anything other than "True" (case-insensitive)


# Read in variables from the input file; there's probably a cleaner way to do this, but no matter.
for entry in input_file_lines:
    if entry[0] == 'initial_structure':
        initial_structure = entry[2]
    elif entry[0] == 'if_glob':
        if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
            sys.exit('Error: if_glob must be either True or False')
        if_glob = str2bool(entry[2])
    elif entry[0] == 'topology':
        topology = entry[2]
    elif entry[0] == 'n_adjust':
        try:
            if int(entry[2]) < 1:
                sys.exit('Error: n_adjust must be greater than or equal to 1')
        except ValueError:
            sys.exit('Error: n_adjust must be an integer')
        n_adjust = int(entry[2])
    elif entry[0] == 'batch_system':
        if not entry[2].lower() == 'pbs' and not entry[2].lower() == 'slurm':
            sys.exit('Error: batch_system must be either pbs or slurm')
        batch_system = entry[2].lower()
    elif entry[0] == 'working_directory':
        if not '/' in entry[2]: # interpreting as a subfolder of cwd
            working_directory = os.getcwd() + '/' + entry[2]
        else:                   # interpreting as an absolute path
            working_directory = entry[2]
    elif entry[0] == 'commit_fwd':
        if ' ' in entry[2]:
            sys.exit('Error: commit_define_fwd cannot contain whitespace (\' \') characters')
        commit_define_fwd = ast.literal_eval(entry[2])  # converts a string looking like a list, to an actual list
        if len(commit_define_fwd) < 4:
            sys.exit('Error: commit_fwd must have four rows')
    elif entry[0] == 'commit_bwd':
        if ' ' in entry[2]:
            sys.exit('Error: commit_define_bwd cannot contain whitespace (\' \') characters')
        commit_define_bwd = ast.literal_eval(entry[2])
        if len(commit_define_bwd) < 4:
            sys.exit('Error: commit_bwd must have four rows')
    elif entry[0] == 'candidate_op':
        if ' ' in entry[2]:
            sys.exit('Error: candidate_op cannot contain whitespace (\' \') characters')
        candidateops = ast.literal_eval(entry[2])
        if len(candidateops) < 2:
            sys.exit('Error: candidate_op must have at least two rows')
    elif entry[0] == 'restart_on_crash':
        if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
            sys.exit('Error: restart_on_crash must be either True or False')
        restart_on_crash = str2bool(entry[2])
    elif entry[0] == 'max_fails':
        try:
            max_fails = int(entry[2])
        except ValueError:
            sys.exit('Error: max_fails must be an integer')
    elif entry[0] == 'max_moves':
        try:
            max_moves = int(entry[2])
        except ValueError:
            sys.exit('Error: max_moves must be an integer')
    elif entry[0] == 'max_accept':
        try:
            max_accept = int(entry[2])
        except ValueError:
            sys.exit('Error: max_accept must be an integer')
    elif entry[0] == 'degeneracy':
        try:
            if int(entry[2]) < 1:
                sys.exit('Error: degeneracy must be greater than or equal to 1')
        except ValueError:
            sys.exit('Error: degeneracy must be an integer')
        degeneracy = int(entry[2])
    elif entry[0] == 'init_nodes':
        try:
            if int(entry[2]) < 1:
                sys.exit('Error: init_nodes must be greater than or equal to 1')
        except ValueError:
            sys.exit('Error: init_nodes must be an integer')
        init_nodes = int(entry[2])
    elif entry[0] == 'init_ppn':
        try:
            if int(entry[2]) < 1:
                sys.exit('Error: init_ppn must be greater than or equal to 1')
        except ValueError:
            sys.exit('Error: init_ppn must be an integer')
        init_ppn = int(entry[2])
    elif entry[0] == 'init_walltime':
        if not ([len(num) for num in entry[2].split(':')] == [2,2,2]):
            sys.exit('Error: init_walltime must have format: HH:MM:SS')
        try:
            [int(num) for num in entry[2].split(':')]
        except ValueError:
            sys.exit('Error: init_walltime must consist only of colon-separated integers')
        init_walltime = entry[2]
    elif entry[0] == 'prod_nodes':
        try:
            if int(entry[2]) < 1:
                sys.exit('Error: prod_nodes must be greater than or equal to 1')
        except ValueError:
            sys.exit('Error: prod_nodes must be an integer')
        prod_nodes = int(entry[2])
    elif entry[0] == 'prod_ppn':
        try:
            if int(entry[2]) < 1:
                sys.exit('Error: prod_ppn must be greater than or equal to 1')
        except ValueError:
            sys.exit('Error: prod_ppn must be an integer')
        prod_ppn = int(entry[2])
    elif entry[0] == 'prod_walltime':
        if not ([len(num) for num in entry[2].split(':')] == [2,2,2]):
            sys.exit('Error: prod_walltime must have format: HH:MM:SS')
        try:
            [int(num) for num in entry[2].split(':')]
        except ValueError:
            sys.exit('Error: prod_walltime must consist only of colon-separated integers')
        prod_walltime = entry[2]
    elif entry[0] == 'resample':
        if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
            sys.exit('Error: resample must be either True or False')
        resample = str2bool(entry[2])
    elif entry[0] == 'fork':
        try:
            if int(entry[2]) < 1:
                sys.exit('Error: fork must be greater than or equal to 1')
        except ValueError:
            sys.exit('Error: fork must be an integer')
        fork = int(entry[2])
    elif entry[0] == 'home_folder':
        home_folder = entry[2]
    elif entry[0] == 'always_new':
        if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
            sys.exit('Error: always_new must be either True or False')
        always_new = str2bool(entry[2])
    elif entry[0] == 'rc_definition':
        rc_definition = entry[2]
    elif entry[0] == 'rc_minmax':
        if ' ' in entry[2]:
            sys.exit('Error: rc_minmax cannot contain whitespace (\' \') characters')
        rc_minmax = ast.literal_eval(entry[2])
        if rc_minmax:   # to allow rc_minmax = '', only perform these checks if it's not
            if not len(rc_minmax) == 2:
                sys.exit('Error: rc_minmax must have two rows')
            for i in range(len(rc_minmax)):
                if rc_minmax[0][i] and rc_minmax[1][i] and rc_minmax[0][i] >= rc_minmax[1][i]:
                    sys.exit('Error: values in the second row of rc_minmax must be larger than the corresponding values in the first row')
    elif entry[0] == 'committor_analysis':
        committor_analysis = ast.literal_eval(entry[2])
        if committor_analysis:  # to allow committor_analysis = [], only perform these checks if it's not
            if not len(committor_analysis) == 5:
                sys.exit('Error: committor_analysis must be of length five (yours is of length ' + str(len(committor_analysis)) + ')')
            for i in range(len(committor_analysis)):
                if i in [1,2] and type(committor_analysis[i]) not in [float,int]:
                    sys.exit('Error: committor_analysis[' + str(i) + '] must have type float or int, but has type: ' + str(type(committor_analysis[i])))
                elif i in [0,3,4] and type(committor_analysis[i]) not in [int]:
                    sys.exit('Error: committor_analysis[' + str(i) + '] must have type int, but has type: ' + str(type(committor_analysis[i])))
    elif entry[0] == 'restart':
        if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
            sys.exit('Error: restart must be either True or False')
        restart = str2bool(entry[2])

# Initialize jinja2 environment for filling out templates
if os.path.exists(home_folder + '/' + 'templates'):
    env = Environment(
        loader=FileSystemLoader(home_folder + '/' + 'templates'),
    )
else:
    sys.exit(
        'Error: could not locate templates folder: ' + home_folder + '/' + 'templates\nSee documentation for the'
                                                                               '\'home_folder\' option.')

# Return an error and exit if the input file is missing entries for non-optional inputs.
if 'commit_fwd' not in [entry[0] for entry in input_file_lines] and not resample and not rc_definition:
    sys.exit('Error: input file is missing entry for commit_fwd, which is non-optional for this standard AS run')
if 'commit_bwd' not in [entry[0] for entry in input_file_lines] and not resample and not rc_definition:
    sys.exit('Error: input file is missing entry for commit_bwd, which is non-optional for this standard AS run')
if 'candidate_op' not in [entry[0] for entry in input_file_lines]:
    sys.exit('Error: input file is missing entry for candidate_op, which is non-optional for this standard AS run')

if restart and (resample or rc_definition or committor_analysis):
    problem = ''
    if resample:
        problem = 'resample = True'
    elif rc_definition:
        problem = 'rc_definition is defined'
    elif committor_analysis:
        problem = 'committor_analysis is defined'
    sys.exit('Error: the following options are incompatible: restart = True and ' + problem)
elif restart:
    # First, carefully load in the necessary information:
    try:
        os.chdir(working_directory)
    except FileNotFoundError:
        sys.exit('Error: restart = True, but I cannot find the working directory: ' + working_directory)
    try:
        allthreads = pickle.load(open('restart.pkl', 'rb'))
    except FileNotFoundError:
        sys.exit('Error: restart = True, but I cannot find restart.pkl inside working directory: ' + working_directory)
    running = []
    itinerary = []
    # Next, add those threads that haven't terminated to the itinerary and call main_loop
    for thread in allthreads:
        if thread.status not in ['max_accept', 'max_moves', 'max_fails']:
            itinerary.append(thread)
    main_loop()
    sys.exit()


if rc_definition and not committor_analysis:
    rc_eval.return_rcs(candidateops,rc_definition,working_directory,topology,rc_minmax)
    sys.exit('\nCompleted reaction coordinate evaluation run. See ' + working_directory + '/rc_eval.out for results.')
elif rc_definition and committor_analysis:
    rc_eval.committor_analysis(candidateops,rc_definition,working_directory,topology,rc_minmax,committor_analysis,batch_system)
    sys.exit('\nCompleted committor analysis run. See ' + working_directory + '/committor_analysis.out for results.')
elif not rc_definition and committor_analysis:
    sys.exit('Error: committor analysis run requires rc_definition to be defined')

if if_glob:
    start_name = glob.glob(initial_structure)   # list of names of coordinate files to begin shooting from
else:
    start_name = [initial_structure]

for filename in start_name:
    if ' ' in filename:
        sys.exit('Error: one or more input coordinate filenames contains a space character, which is not supported\n'
                 'This first offending filename found was: ' + filename)

if len(start_name) == 0:
    sys.exit('Error: no initial structure found. Check input options initial_structure and if_glob.')
for init in start_name:
    if not os.path.exists(init):
        sys.exit('Error: could not find initial structure file: ' + init)

if degeneracy > 1:
    temp = []                   # initialize temporary list to substitute for start_name later
    for init in start_name:
        for i in range(degeneracy):
            shutil.copy(init, init + '_' + str(i+1))
            temp.append(init + '_' + str(i+1))
    start_name = temp

# Make a working directory
dirName = working_directory
if not resample and overwrite:  # if resample == True, we want to keep our old working directory
    if os.path.exists(dirName):
        shutil.rmtree(dirName)  # delete old working directory
    os.makedirs(dirName)        # make a new one
elif resample:
    if not os.path.exists(dirName):
        sys.exit('Error: resample = True, but I can\'t find the working directory: ' + dirName)
elif not overwrite:             # if resample == False and overwrite == False, make sure dirName doesn't exist...
    if os.path.exists(dirName):
        sys.exit('Error: overwrite = False, but working directory ' + dirName + ' already exists. Move it, choose a '
                 'different working directory, or add option -O to overwrite it.')
    else:
        os.makedirs(dirName)

os.chdir(working_directory)     # move to working directory

itinerary = []                  # a list of threads that need running
running = []                    # a list of currently running threads
allthreads = []                 # a list of all threads regardless of status

for structure in start_name:                # for all of the initial structures...
    thread = spawnthread(structure,suffix='1')              # spawn a new thread with the default settings
    allthreads.append(thread)                               # add it to the list of all threads for bookkeeping
    thread.last_valid = '0'                                 # so that if the first shooting point does not result in a valid transition path, shooting will begin from the TS guess
    thread.prmtop = topology                                # set prmtop filename for the thread
    itinerary.append(thread)                                # submit it to the itinerary
    shutil.copy(called_path + '/' + structure, './')        # and copy the input structure to the working directory...
    shutil.copy(called_path + '/' + thread.prmtop, './')    # ... and its little topology file, too!
    if degeneracy > 1:                                      # if degeneracy > 1, files in start_name were copies...
        os.remove(called_path + '/' + structure)            # ... delete them to keep the user's space clean!

if not resample:
    try:
        os.remove('as.log')                 # delete previous run's log
    except OSError:                         # catches error if no previous log file exists
        pass
    with open('as.log', 'w+') as newlog:
        newlog.write('New log file')        # make new log file
        newlog.close()

try:
    os.remove('as.out')                 # delete previous run's output file
except OSError:                         # catches error if no previous output file exists
    pass
with open('as.out', 'w+') as newout:    # make a new output file
    newout.close()

if resample:                            # if True, this is a resample run, so we'll head off the simulations steps here
    pattern = re.compile('\ .*\ finished')                              # pattern to find job name
    pattern2 = re.compile('result:\ [a-z]*\ ')                          # pattern for basin commitment flag
    try:
        logfile = open('as.log')                                        # open log for reading...
    except OSError:
        sys.exit('Error: could not find as.log in working directory: ' + working_directory)
    logfile_lines = logfile.readlines()
    for line in logfile_lines:                                          # iterate through log file
        if 'finished with fwd trajectory result: ' in line:             # looking for lines with results
            commit = pattern2.findall(line)[0][8:-1]                    # first, identify the commitment flag
            if commit != 'fail':                                        # none of this matters if it was "fail"
                basin = 'error'
                if commit == 'fwd':
                    basin = 'A'
                elif commit == 'bwd':
                    basin = 'B'
                init_name = pattern.findall(line)[0][5:-9] + '_init_fwd.rst' # clunky
                prmtop = topology
                fakethread = Thread()
                fakethread.start_name = init_name                       # making a fake thread for candidatevalues
                fakethread.prmtop = prmtop
                open('as.out', 'a').write(basin + ' <- ' + candidatevalues(fakethread) + '\n') # todo: test this line of resample
    sys.exit('Resampling complete; written new as.out')

os.makedirs('history')                  # make a new directory to contain the history files of each thread

main_loop()
