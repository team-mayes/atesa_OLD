# Code to perform evaluation of reaction coordinate equation for each shooting point and output the results, sorted by
# RC closest to zero and local acceptance ratio maximized and descending, to working_directory/rc_eval.out

# iterate through each shooting point coordinate file, using pytraj to evaluate OPs and performing the requested calculations on them as defined by rc_definition
# assemble into a list, sort, and output to rc_eval.out
# use eval(rc_definition[1]) to interpret arbitrary strings in rc_definition[1] as python code.
# before doing that, replace OP strings with their appropriate values as reported by pytraj!

from __future__ import division     # causes division to work properly when using Python 2 todo: test this
import os
import pytraj
import glob
import numpy                        # appears unused, but is required to support numpy functions passed in rc_definition
import sys
import re
import importlib
import subprocess
import time


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
            try:
                if not rc_minmax[0][index] or not rc_minmax[1][index]:  # if there's a blank entry in rc_minmax
                    sys.exit('\nError: rc_definition contains reference to OP' + str(index+1) + ' without a corresponding entry in rc_minmax')
            except IndexError:                                          # if there's no entry at all
                sys.exit('\nError: rc_definition contains reference to OP' + str(index + 1) + ' without a corresponding entry in rc_minmax')
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
    logfile.close()
    for line in logfile_lines:                              # iterate through log file
        if re.match(pattern2, line) is not None:            # looking for lines with coordinate file names
            filename = pattern.findall(line)[0][14:]        # read out file name
            if filename not in filelist:                    # if this one is not redundant...
                filelist.append(filename)                   # append the file name to the list of names

    # Next, we'll interpret the equation encoded in rc_definition and candidateops...
    equation = rc_definition
    for j in reversed(range(len(candidateops[0]))):         # for each candidate op... (reversed to avoid e.g. 'OP10' -> 'candidatevalues(..., 0)0')
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
        results = sorted(results,key=lambda x: abs(x[0]))
        for result in results:
            file.write(result[1] + ': ' + str(result[0][0]) + '\n')
        file.close()


def committor_analysis(candidateops,rc_definition,working_directory,prmtop,rc_minmax,committor_analysis_options,batch_system,commit_define_fwd,commit_define_bwd):
    aimless_shooting = importlib.import_module('aimless_shooting')  # import here to avoid issues caused by circular import statements

    def checkcommit(name, prmtop, directory=''):
        # Copied directly from aimless_shooting.py, with minor edits to make it standalone.

        committor_directory = ''
        if directory:
            directory += '/'
            committor_directory = directory + 'committor_analysis/'

        if not os.path.isfile(committor_directory + name):  # if the file doesn't exist yet, just do nothing
            return ''

        traj = pytraj.iterload(committor_directory + name, directory + prmtop, format='.nc')

        if not traj:  # catches error if the trajectory file exists but has zero frames
            print(
                'Don\'t worry about this internal error; it just means that aimless_shooting is checking for commitment in a trajectory that doesn\'t have any frames yet, probably because the simulation has only just begun.')
            return ''

        commit_flag = ''  # initialize flag for commitment; this is the value to be returned eventually
        for i in range(0, len(commit_define_fwd[2])):
            if commit_define_fwd[3][i] == 'lt':
                if pytraj.distance(traj, commit_define_fwd[0][i] + ' ' + commit_define_fwd[1][i], n_frames=1)[-1] <= \
                        commit_define_fwd[2][i]:
                    commit_flag = 'fwd'  # if a commitor test is passed, testing moves on to the next one.
                else:
                    commit_flag = ''
                    break  # if a commitor test is not passed, all testing in this direction ends in a fail
            elif commit_define_fwd[3][i] == 'gt':
                if pytraj.distance(traj, commit_define_fwd[0][i] + ' ' + commit_define_fwd[1][i], n_frames=1)[-1] >= \
                        commit_define_fwd[2][i]:
                    commit_flag = 'fwd'
                else:
                    commit_flag = ''
                    break
            else:
                open('as.log', 'a').write(
                    '\nAn incorrect commitor definition \"' + commit_define_fwd[3][i] + '\" was given for index ' + str(
                        i) + ' in the \'fwd\' direction.')
                sys.exit(
                    'An incorrect commitor definition \"' + commit_define_fwd[3][i] + '\" was given for index ' + str(
                        i) + ' in the \'fwd\' direction.')

        if commit_flag == '':  # only bother checking for bwd commitment if not fwd commited
            for i in range(0, len(commit_define_bwd[2])):
                if commit_define_bwd[3][i] == 'lt':
                    if pytraj.distance(traj, commit_define_bwd[0][i] + ' ' + commit_define_bwd[1][i], n_frames=1)[-1] <= \
                            commit_define_bwd[2][i]:
                        commit_flag = 'bwd'  # if a commitor test is passed, testing moves on to the next one.
                    else:
                        commit_flag = ''
                        break  # if a commitor test is not passed, all testing in this direction ends in a fail
                elif commit_define_bwd[3][i] == 'gt':
                    if pytraj.distance(traj, commit_define_bwd[0][i] + ' ' + commit_define_bwd[1][i], n_frames=1)[-1] >= \
                            commit_define_bwd[2][i]:
                        commit_flag = 'bwd'
                    else:
                        commit_flag = ''
                        break
                else:
                    open('as.log', 'a').write('\nAn incorrect commitor definition \"' + commit_define_bwd[3][
                        i] + '\" was given for index ' + str(i) + ' in the \'bwd\' direction.')
                    sys.exit('An incorrect commitor definition \"' + commit_define_bwd[3][
                        i] + '\" was given for index ' + str(i) + ' in the \'bwd\' direction.')

        del traj  # to ensure cleanup of iterload object

        return commit_flag

    try:
        os.remove('ca.log')                 # delete previous run's log file
    except OSError:                         # catches error if no previous log file exists
        pass
    with open('ca.log', 'w+') as newlog:    # make a new log file
        newlog.write('New log file')
        newlog.close()
        logfile = os.getcwd() + '/ca.log'

    # Implementation of committor analysis. Start by obtaining the list of RC values to work with...
    if not os.path.exists(working_directory + '/rc_eval.out'):
        open('ca.log', 'a').write('\nNo rc_eval.out found in working directory, generating it...')
        return_rcs(candidateops,rc_definition,working_directory,prmtop,rc_minmax)
    else:
        open('ca.log', 'a').write('\nFound ' + working_directory + '/rc_eval.out, continuing...')
    open('ca.log', 'a').close()

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

    try:
        os.makedirs('committor_analysis')
    except FileExistsError:
        pass
    os.chdir('committor_analysis')

    eligible = []                               # initialize list of eligible shooting points for committor analysis
    eligible_rcs = []                           # another list holding corresponding rc values
    lines = open('../rc_eval.out', 'r').readlines()
    open('../rc_eval.out', 'r').close()
    for line in lines:
        splitline = line.split(' ')             # split line into list [shooting point filename, rc value]
        if abs(float(splitline[1]) - rc_ts) <= rc_tol:
            eligible.append(splitline[0][:-1])  # [:-1] removes trailing colon
            eligible_rcs.append(splitline[1])

    if len(eligible) == 0:
        sys.exit('Error: attempted committor analysis, but couldn\'t find any shooting points with reaction coordinate '
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
                    history_lines = open('../history/' + history_filename, 'r').readlines()
                    open('../history/' + history_filename, 'r').close()
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
        thread.prmtop = prmtop
        threadlist.append(thread)

    running = []
    for thread in threadlist:
        aimless_shooting.makebatch(thread,n_shots)
        thread.jobidlist = aimless_shooting.subbatch(thread,n_shots=n_shots,logfile=logfile)
        running.append(thread)
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

    # todo: debug this; it submits the jobs correctly, but then exits without error very early on, while running should still contain most threads... possibly "jobid not in str(output)" is returning True errantly?
    while running:
        process = subprocess.Popen(queue_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, close_fds=True, shell=True)
        output = process.stdout.read()
        # The above line retrieves both the stdout and stderr streams into the same variable; on PBS, sometimes this
        # returns a "busy" message. The following while loop is meant to handle that, but it's obviously ugly.
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
                    this_commit = aimless_shooting.checkcommit(thread.basename + '_ca_' + str(number) + '.nc',
                                                               thread.prmtop, working_directory)
                    if not this_commit:
                        thread.commitlist[number] = 'fail'
                    else:
                        thread.commitlist[number] = this_commit
                    thread.jobidlist[number] = 'skip'
                elif not jobid == 'skip':                               # job is still running
                    this_commit = aimless_shooting.checkcommit(thread.basename + '_ca_' + str(number) + '.nc',
                                                               thread.prmtop, working_directory)
                    if this_commit:
                        thread.commitlist[number] = this_commit
                        process = subprocess.Popen([cancel_command, thread.jobidlist[number]], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()           # doesn't do anything, I think
                        thread.jobidlist[number] = 'skip'
                elif jobid == 'skip':                                   # job has already been handled
                    skipcount += 1
            if skipcount == n_shots:                # if all the jobs associated with this thread have ended...
                ### This block of code originally meant to provide partial output as committor_analysis runs.
                # with open('committor_analysis.out', 'a') as outputfile: # write to output as we're going for convenience
                #     for commit in running[index].commitlist:
                #         if commit != 'fail':
                #             outputfile.write(commit + ' ' + eligible_rcs[threadlist.index(running[index])])
                #     outputfile.close()
                del running[index]
                index -= 1
            index += 1

        time.sleep(60)                              # just wait a bit so as not to overwhelm the batch system

    # Finally, we just need to interpret the results and output them in a format that's convenient for a histogram
    # todo: this code currently rechecks every trajectory for commitment, rather than using the values stored in each thread.commitlist. Fix this!
    trajectories = sorted(glob.glob('*_ca_*.nc'))

    with open('committor_analysis.out', 'w') as f:
        f.close()

    last_basename = ''
    this_committor = []
    count = 0
    for trajectory in trajectories:
        count += 1
        update_progress(count / len(trajectories))
        basename = trajectory[:-8]  # name excluding '_ca_#.nc' where # is one digit (only works because mine are 0-9)
        commit = checkcommit(trajectory, 'ts_guess.prmtop', working_directory)  # todo: the offending line for the above todo (other lines may also need to change of course)
        if basename != last_basename or trajectory == trajectories[-1]:
            if trajectory == trajectories[-1]:  # sloppy, but whatever
                this_committor.append(commit)
            # collate results of previous basename
            fwd_count = 0
            bwd_count = 0
            if this_committor:  # true for every pass except first one
                for result in this_committor:
                    if result == 'fwd':
                        fwd_count += 1
                    elif result == 'bwd':
                        bwd_count += 1
                try:
                    pb = fwd_count / (fwd_count + bwd_count)
                    open('committor_analysis.out', 'a').write(str(pb) + '\n')
                    open('committor_analysis.out', 'a').close()
                except ZeroDivisionError:
                    pass
            # start new basename
            last_basename = basename
            this_committor = [commit]
        else:  # if basename == last_basename
            this_committor.append(commit)


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
