# Code to perform evaluation of reaction coordinate equation for each shooting point and output the results, sorted by
# RC closest to zero and local acceptance ratio maximized and descending, to working_directory/rc_eval.out

# iterate through each shooting point coordinate file, using pytraj to evaluate OPs and performing the requested calculations on them as defined by rc_definition
# assemble into a list, sort, and output to rc_eval.out
# use eval(rc_definition[1]) to interpret arbitrary strings in rc_definition[1] as python code.
# before doing that, replace OP strings with their appropriate values as reported by pytraj!

import os
import pytraj
import numpy    # appears unused, but is required to support numpy functions passed in rc_definition
import sys
import re
import importlib
import subprocess
import time
aimless_shooting = importlib.import_module('aimless_shooting')

def return_rcs(candidateops,rc_definition,working_directory,prmtop,rc_minmax):
    # First, copy the definition of candidatevalues from aimless_shooting.py, modified to be called with a pytraj
    # trajectory object (or really a single frame) rather than with a thread object.
    def candidatevalues(coord_file, index):
        # Returns a string containing the index-th candidate OP value defined by the user in the candidateops object,
        # reduced based on the contents of rc_minmax
        traj = pytraj.iterload(coord_file, prmtop)

        if len(candidateops) == 4:  # candidateops contains dihedrals
            if candidateops[3][index]:  # if this OP is a dihedral
                value = pytraj.dihedral(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' +
                                                   candidateops[2][index] + ' ' + candidateops[3][index])
            elif candidateops[2][index]:  # if this OP is an angle
                value = pytraj.angle(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' +
                                                candidateops[2][index])
            else:  # if this OP is a distance
                value = pytraj.distance(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index])
            output = value
        elif len(candidateops) == 3:  # candidateops contains angles but not dihedrals
            if candidateops[2][index]:  # if this OP is an angle
                value = pytraj.angle(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' +
                                                candidateops[2][index])
            else:  # if this OP is a distance
                value = pytraj.distance(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index])
            output = value
        else:  # candidateops contains only distances
            value = pytraj.distance(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index])
            output = value

        # Before returning the result, we want to convert to a reduced variable z = (r-rmin)/(rmax-rmin)
        if rc_minmax:
            if not rc_minmax[0][index] or not rc_minmax[1][index]:
                sys.exit('\nError: rc_definition contains reference to OP' + str(index+1) + ' without a corresponding entry in rc_minmax')
            output = (output - rc_minmax[0][index])/(rc_minmax[1][index] - rc_minmax[0][index])

        return output


    try:
        os.chdir(working_directory)
    except FileNotFoundError:
        sys.exit('Error: could not find working directory: ' + working_directory)

    # First, we want to assemble a list of all the shooting point coordinate file names
    filelist = []                                           # initialize list of filenames to evaluate

    pattern = re.compile('starting\ from\ .*')              # pattern to find shooting point coordinate file name
    pattern2 = re.compile('^Writing init batch file for ')  # pattern to find appropriate lines in logfile
    try:
        logfile = open('as.log')  # open log for reading...
    except OSError:
        sys.exit('Error: could not find as.log in working directory: ' + working_directory)
    logfile_lines = logfile.readlines()
    for line in logfile_lines:                              # iterate through log file
        if re.match(pattern2, line) is not None:            # looking for lines with coordinate file names
            filename = pattern.findall(line)[0][14:]        # read out file name
            if filename not in filelist:                    # if this one is not redundant...
                filelist.append(filename)                   # append the file name to the list of names

    # Next, we'll interpret the equation encoded in rc_definition and candidateops...
    equation = rc_definition
    for j in range(len(candidateops[0])):  # for each candidate op...
        equation = equation.replace('OP' + str(j+1), 'candidatevalues(\'coord_file\',' + str(j) + ')')

    # Finally, evaluate and print to output for each coordinate file.
    with open('rc_eval.out', 'w') as file:
        results = []                                        # initialize list of RC values
        index = 0
        for item in filelist:
            update_progress(index/len(filelist))
            this_equation = equation.replace('coord_file',item)
            results.append([eval(this_equation), item])
            index += 1
        update_progress(index / len(filelist))
        results = sorted(results,key=lambda x: x[0])
        for result in results:
            file.write(result[1] + ': ' + str(result[0][0]) + '\n')


def committor_analysis(candidateops,rc_definition,working_directory,prmtop,rc_minmax,committor_analysis_options,batch_system):
    # Implementation of committor analysis. Start by calling return_rcs to get list of RC values to work with...
    return_rcs(candidateops,rc_definition,working_directory,prmtop,rc_minmax)

    # Import arguments from user's input file
    n_shots = committor_analysis_options[0]
    rc_ts = committor_analysis_options[1]
    rc_tol = committor_analysis_options[2]
    min_points = committor_analysis_options[3]
    min_dist = committor_analysis_options[4]

    try:
        os.chdir(working_directory)
    except FileNotFoundError:
        sys.exit('Error: could not find working directory: ' + working_directory)

    os.makedirs('committor_analysis')
    os.chdir('committor_analysis')

    eligible = []                               # initialize list of eligible shooting points for committor analysis
    eligible_rcs = []                           # another list holding corresponding rc values
    lines = open('../rc_eval.out', 'r').readlines()
    for line in lines:
        splitline = line.split(' ')             # split line into list [shooting point filename, rc value]
        if abs(float(splitline[1]) - rc_ts) <= rc_tol:
            eligible.append(splitline[0])
            eligible_rcs.append(splitline[1])

    if len(eligible) == 0:
        sys.exit('Error: attempted comittor analysis, but couldn\'t find any shooting points with reaction coordinate '
                 'values within ' + str(rc_tol) + ' of ' + str(rc_ts) + ' in working directory: ' + working_directory)

    # Next I want to remove all the violations of the min_dist setting. To do this, I'll scan history files whose names
    # are contained within each line of eligible for the indices of filenames that appear in eligible, and then remove
    # the one with the larger index from eligible if there are any conflicts.
    #
    # First I want to define a helper function to remove items within min_dist of one another from eligible, based on a
    # list of indices within history_lines corresponding to lines that are within eligible. This function breaks and
    # returns True every time it removes an element from the two lists, so that we can call it in a loop with the
    # contents of the lists updated at each attempt. todo: test this
    def handle_min_dist(indices,history_lines):
        if len(indices) >= 2:
            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] < min_dist:
                    try:
                        del_index = eligible.index(history_lines[indices[i + 1]].split(' ')[0])
                        eligible[del_index] = ''
                        eligible_rcs[del_index] = ''
                        eligible.remove('')
                        eligible_rcs.remove('')
                        del indices[i + 1]
                        return True
                    except ValueError:          # this is super complicated... just in case it breaks, do this:
                        sys.exit('Error: The developer broke something in rc_eval.committor_analysis.handle_min_dist().'
                                 'Please email tburgin@umich.edu, or report this issue on GitHub!!')
        return False

    if min_dist > 1:                            # only bother if min_dist is greater than one
        for filename in eligible:
            for history_filename in os.listdir('../history'):
                if history_filename in filename:
                    history_lines = open(history_filename, 'r').readlines()
                    indices = []                # initialize indices of matches
                    index = 0                   # initialize index to keep track
                    for history_line in history_lines:
                        if history_line.split(' ')[0] in eligible:
                            indices.append(index)
                        index += 1
                    cont = True
                    while cont:                 # to keep my indices from getting tangled up by deletions
                        cont = handle_min_dist(indices,history_lines)

    # At this point we have a list (eligible) of shooting points. Next step is to count it, do some constrained sampling
    # to make up the difference between the count and min_points, and then let each of them go n_shots times and record
    # the results into the output file.

    ### Insert constrained sampling code here ###
    # Should result in new files being created in working_directory/committor_analysis, and the relevant names being
    # appended to eligible

    # Finally, we want to submit n_shots jobs for each coordinate file in eligible.
    # To do this, I'll hijack the thread system from the main code...
    threadlist = []
    for point in eligible:
        thread = aimless_shooting.spawnthread(point,type='committor_analysis',suffix='0')
        thread.prmtop = '../' + prmtop
        threadlist.append(thread)

    running = []
    for thread in threadlist:
        aimless_shooting.makebatch(thread,n_shots)
        jobids = aimless_shooting.subbatch(thread,n_shots=n_shots)
        running.append(thread)
        # Importantly, the order of the jobids in this string corresponds to the order in which they were created, which
        # means that e.g., jobids[3] corresponds to the job that produces the trajectory named:
        # thread.basename + '_ca_3' + '.nc'
        for jobid in jobids:
            thread.jobidlist.append(jobid)
        thread.commitlist = [[] for dummy in range(n_shots)]    # initialize commitlist to contain n_shots values

    # Having submitted the jobs to the batch system, now all I need to do is track their results as in the main code.
    # A little bit of redundancy here from the main code, but it's still cleaner than doing this all in the same file.
    user_alias = '$USER'
    if batch_system == 'pbs':
        queue_command = 'qselect -u ' + user_alias + ' -s QR'
        cancel_command = 'qdel'
    elif batch_system == 'slurm':
        queue_command = 'squeue -u ' + user_alias
        cancel_command = 'scancel'
    else:
        sys.exit('Error: incorrect batch system type: ' + batch_system)

    # todo: test this
    while running:
        process = subprocess.Popen(queue_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, close_fds=True,
                                   shell=True)      # check list of currently running jobs
        # The above line retrieves both the stdout and stderr streams into the same variable; on PBS, sometimes this
        # returns a "busy" message. The following while loop is meant to handle that, but it's obviously ugly.
        output = process.stdout.read()
        while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
            process = subprocess.Popen(queue_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, close_fds=True, shell=True)
            output = process.stdout.read()

        index = 0  # set place in running to 0
        while index < len(running):                 # while instead of for to control indexing manually
            thread = running[index]
            number = -1
            skipcount = 0
            for jobid in thread.jobidlist:
                number += 1                         # this is the number in range(n_shots) for this jobid
                if not jobid == 'skip' and jobid not in str(output):    # job terminated on its own
                    this_commit = aimless_shooting.checkcommit(thread, 'ca_' + number)
                    if not this_commit:
                        thread.commitlist[number] = 'fail'
                    else:
                        thread.commitlist[number] = this_commit
                    thread.jobidlist[number] = 'skip'
                elif not jobid == 'skip':                               # job is still running
                    this_commit = aimless_shooting.checkcommit(thread, 'ca_' + number)
                    if this_commit:
                        thread.commitlist[number] = this_commit
                        process = subprocess.Popen([cancel_command, thread.jobidlist[number]], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()           # doesn't do anything, I think
                        thread.jobidlist[number] = 'skip'
                elif jobid == 'skip':                                   # job has already been handled
                    skipcount += 1
            if skipcount == n_shots:                # if all the jobs associated with this thread have ended...
                del running[index]
                index -= 1
            index += 1

        time.sleep(60)                              # just wait a bit so as not to overwhelm the batch system

    # Finally, we just need to interpret the results and output them in a format that's convenient for a histogram
    with open('committor_analysis.out','w') as outputfile:
        index = -1
        for thread in threadlist:
            index += 1
            for commit in thread.commitlist:
                if commit != 'fail':
                    # Write commit value, a space, and then the reaction coordinate value from eligible_rcs
                    # This works because the order of threads in threadlist corresponds to the order of eligible_rcs
                    outputfile.write(commit + ' ' + eligible_rcs[index])


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done!\r\n"
    block = int(round(barLength*progress))
    text = "\rCalculating reaction coordinate values: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()
