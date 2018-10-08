# Code to perform evaluation of reaction coordinate equation for each shooting point and output the results, sorted by
# RC closest to zero and local acceptance ratio maximized and descending, to working_directory/rc_eval.out

# iterate through each shooting point coordinate file, using pytraj to evaluate OPs and performing the requested calculations on them as defined by rc_definition
# assemble into a list, sort, and output to rc_eval.out
# use eval(rc_definition[1]) to interpret arbitrary strings in rc_definition[1] as python code.
# before doing that, replace OP strings with their appropriate values as reported by pytraj!

from __future__ import division     # causes division to work properly when using Python 2
import os
import pytraj
import glob
import numpy                        # appears unused, but is required to support numpy functions passed in rc_definition
import sys
import re
import importlib
import subprocess
import time
import shutil
import fileinput
import math


def return_rcs(**kwargs):
    """
    Produce rc_eval.out, a sorted list of RC values for each shooting point, in the working directory.

    This function can be called standalone by atesa.py when rc_definition is given by committor_analysis is
    not; alternatively, it can be called by committor_analysis. All the parameters it takes are simply passed through
    from variables of the same names in atesa.py.

    Parameters
    ----------
    candidateops : list
        The parsed list of user-defined candidate OPs.
    rc_definition : str
        The user-defined reaction coordinate equation, containing OP[X] (where X is an integer) for each OP value.
    working_directory : str
        Path to the aimless shooting working directory (contains shooting point coordinate files, as.log, etc.)
    prmtop : str
        Name of the parameter/topology file for this run.
    rc_minmax : list
        List of format [mins,maxes] where mins and maxes are both lists containing the minimum and maximum observed
        values of each OP, in ascending order, respectively.
    literal_ops : bool
        Boolean identifying whether candidateops should be interpreted as a list of literal strings to be evaluated
        using eval() (True), or whether the definitions are given using the nested list format.

    Returns
    -------
    None

    """

    try:                                    # import here to avoid circular imports
        from atesa import atesa
    except ImportError:                     # atesa is called as a script instead of installed as a package
        atesa = importlib.import_module('atesa')

    def candidatevalues(coord_file, index):
        """
        Evaluate the index'th candidate OP values from the coordinates given by the coord_file

        This code is copied from the atesa.py function of the same name, and modified to accept a single
        coordinate file rather than a thread as its argument, as well as to return only the index'th OP as a float
        rather than every OP as a space-separated string.

        This function may appear "unused"; this is because it is only actually called indirectly within eval() calls. It
        is necessary!

        This function is only capable of returning order parameter rate of change values when the literal_ops Boolean is
        True. This will change in future versions.

        Parameters
        ----------
        coord_file : str
            The name of the coordinate file from which the candidate OP should be read.
        index : int
            The zero-indexed index corresponding to the desired OP.

        Returns
        -------
        float
            The evaluation of the desired candidate OP.

        """
        def increment_coords(coord_file):
            # Modified from revvels() to increment coordinate values by velocities, rather than reversing velocities.
            # Returns the name of the newly-created coordinate file
            byline = open(coord_file).readlines()
            pattern = re.compile('[-0-9.]+')            # regex to match numbers including decimals and negatives
            pattern2 = re.compile('\s[-0-9.]+')         # regex to match numbers including decimals and negatives, with one space in front
            n_atoms = pattern.findall(byline[1])[0]     # number of atoms indicated on second line of .rst file

            shutil.copyfile(coord_file, 'temp' + committor_suffix + '.rst')
            for i, line in enumerate(fileinput.input('temp' + committor_suffix + '.rst', inplace=1)):
                if int(n_atoms) / 2 + 2 > i >= 2:
                    newline = line
                    coords = pattern2.findall(newline)  # line of coordinates
                    vels = pattern2.findall(byline[i + int(math.ceil(int(n_atoms) / 2))])  # corresponding velocities
                    for index in range(len(coords)):
                        length = len(coords[index])     # length of string representing this coordinate
                        replace_string = ' ' + str(float(coords[index]) + float(vels[index]))[0:length - 1]
                        while len(replace_string) < length:
                            replace_string += '0'
                        newline = newline.replace(coords[index], replace_string)
                    sys.stdout.write(newline)
                else:
                    sys.stdout.write(line)

            return 'temp' + committor_suffix + '.rst'

        try:
            traj = pytraj.iterload(coord_file, topology)
        except ValueError:
            sys.exit('Error: coordinate file name ' + coord_file + ' is invalid.')

        if literal_ops:
            try:                            # assuming this is not a rate of change OP...
                output = float(eval(candidateops[0][index]))
            except IndexError:              # this is a rate of change OP after all, so...
                v_0 = float(eval(candidateops[0][int(index - len(candidateops[0]))]))  # value of relevant OP at t = 0
                traj = pytraj.iterload(increment_coords(coord_file), topology)      # new traj for evaluation of OPs
                v_1 = float(eval(candidateops[0][int(index - len(candidateops[0]))]))  # value of OP at t = 1/20.455 ps
                output = float(v_1 - v_0)   # subtract value of op from value 1/20.455 ps earlier

        else:
            if len(candidateops) == 4:  # candidateops contains dihedrals
                if candidateops[3][index]:  # if this OP is a dihedral
                    value = pytraj.dihedral(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' +
                                                       candidateops[2][index] + ' ' + candidateops[3][index])
                elif candidateops[2][index]:  # if this OP is an angle
                    value = pytraj.angle(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' +
                                                    candidateops[2][index])
                else:  # if this OP is a distance
                    value = pytraj.distance(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index])
                output = float(value)
            elif len(candidateops) == 3:  # candidateops contains angles but not dihedrals
                if candidateops[2][index]:  # if this OP is an angle
                    value = pytraj.angle(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' +
                                                    candidateops[2][index])
                else:  # if this OP is a distance
                    value = pytraj.distance(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index])
                output = float(value)
            else:  # candidateops contains only distances
                value = pytraj.distance(traj, mask=candidateops[0][index] + ' ' + candidateops[1][index])
                output = float(value)

        # Before returning the result, we want to convert to a reduced variable z = (r-rmin)/(rmax-rmin)
        if rc_minmax:
            try:
                if not rc_minmax[0][index] or not rc_minmax[1][index]:  # if there's a blank entry in rc_minmax
                    sys.exit('\nError: rc_definition contains reference to OP' + str(index+1) + ' without a corresponding entry in rc_minmax')
            except IndexError:                                          # if there's no entry at all
                sys.exit('\nError: rc_definition contains reference to OP' + str(index + 1) + ' without a corresponding entry in rc_minmax')
            raw_output = output
            output = (output - rc_minmax[0][index])/(rc_minmax[1][index] - rc_minmax[0][index])
            if not -0.01 <= output <= 1.01:
                # For debugging
                # print(raw_output)
                # print(v_0)
                # print(v_1)
                if minmax_error_behavior == 'exit':
                    sys.exit('\nError: reduced variable at index ' + str(index) + ' (zero-indexed) in coordinate file '
                             + coord_file + ' is not between 0 and 1 (value is ' + str(output) + '). '
                             'minmax_error_behavior = exit, so exiting. Check that rc_minmax is correct.')
                elif minmax_error_behavior == 'skip':
                    print('\nWarning: reduced variable at index ' + str(index) + ' (zero-indexed) in coordinate file '
                          + coord_file + ' is not between 0 and 1 (value is ' + str(output) + '). minmax_error_behavior'
                          ' = skip, so this file is being skipped and will not appear in rc_eval' + committor_suffix +
                          '.out')
                    return 'SKIP'
                elif minmax_error_behavior == 'accept':
                    print('\nWarning: reduced variable at index ' + str(index) + ' (zero-indexed) in coordinate file '
                          + coord_file + ' is not between 0 and 1 (value is ' + str(output) + '). minmax_error_behavior'
                          ' = accept, so this file is NOT being skipped and will appear in rc_eval' + committor_suffix +
                          '.out')

        return output

    # Import variables as needed from locals() passed in from atesa.py. This is pretty ugly; the "right" way
    # to do this would have been to have read in variables in the first place inside a function that returns a data
    # structure containing all the necessary variable names. Alternatively, I could simply have written rc_eval inside
    # atesa.py...
    working_directory = kwargs['working_directory']
    rc_definition = kwargs['rc_definition']
    literal_ops = kwargs['literal_ops']
    candidateops = kwargs['candidateops']
    topology = kwargs['topology']
    rc_minmax = kwargs['rc_minmax']
    home_folder = kwargs['home_folder']             # used in atesa.makebatch()
    groupfile = kwargs['groupfile']                 # used in atesa.makebatch()
    env = kwargs['env']                             # used in atesa.makebatch()
    eps_settings = kwargs['eps_settings']           # used in atesa.makebatch()
    include_qdot = kwargs['include_qdot']
    minmax_error_behavior = kwargs['minmax_error_behavior']
    try:    # this key is only present if rc_eval() was called from committor_analysis()
        committor_suffix = kwargs['committor_analysis_options'][5]
    except KeyError:
        committor_suffix = ''

    try:
        os.chdir(working_directory)
    except IOError:
        sys.exit('Error: could not read working directory: ' + working_directory)

    # First, we want to assemble a list of all the shooting point coordinate file names
    filelist = []                                           # initialize list of filenames to evaluate

    # Without doublecheck, there's an error when:
    #   1) One of the jobs for which an init_fwd.rst file was written never finished a prod job, and
    #   2) That shooting point happens to have an OP value outside the range of values observed in the successful jobs
    pattern = re.compile('^Adding\ .*\ forward')            # pattern to find appropriate lines in logfile
    try:
        logfile = open('as.log')                            # open log for reading...
    except OSError:
        sys.exit('\nError: could not find as.log in working directory: ' + working_directory)
    logfile_lines = logfile.readlines()
    logfile.close()
    for line in logfile_lines:                              # iterate through log file
        if re.match(pattern, line) is not None:             # looking for lines with coordinate file names
            filename = pattern.findall(line)[0][7:-8]       # read out thread name
            filename += '_init_fwd.rst'                     # append appropriate suffix to get desired filename
            # Need to double check that the thread has at least one completed shooting move
            doublecheck = False                             # flag indicating status of double checking
            for line in logfile_lines:
                nametocheck = filename[:-13]
                if nametocheck in line and 'finished' in line and 'fwd trajectory result: fail' not in line:
                    doublecheck = True
            if filename not in filelist and doublecheck:    # if this one is not redundant, and has a successful shooting move...
                filelist.append(filename)                   # append the file name to the list of names

    # Next, we'll interpret the equation encoded in rc_definition and candidateops...
    equation = rc_definition
    if literal_ops:
        candidateops = [candidateops]                       # to fix error where candidateops has unexpected format
    if include_qdot:
        qdot_factor = 2                                     # to include qdot OPs if applicable
    else:
        qdot_factor = 1
    for j in reversed(range(int(qdot_factor * len(candidateops[0])))):  # for each candidate op... (reversed to avoid e.g. 'OP10' -> 'candidatevalues(..., 0)0')
        equation = equation.replace('OP' + str(j+1), 'candidatevalues(\'coord_file\',' + str(j) + ')')

    # Finally, evaluate and print to output for each coordinate file.
    # todo: parallelize this somehow? Each item in filelist should be independent, after all. Try dask?
    with open('rc_eval' + committor_suffix + '.out', 'w') as file:
        results = []                                        # initialize list of RC values
        index = 0
        for item in filelist:
            atesa.update_progress(index/len(filelist))
            this_equation = equation.replace('coord_file',item)
            try:
                results.append([eval(this_equation), item])
            except TypeError:   # if minmax_error_behavior = skip and flags this file to be skipped
                print('\nDEBUG: Skipping successfully.')
                pass
            except SyntaxError as e:
                sys.exit('\nError: there was a syntax error in either the candidate_op definitions or in rc_definition.'
                         ' This could be caused by forgetting to specify include_qdot = True in an RC that does in fact'
                         ' include qdot parameters, among other things. Raw exception:\n' + repr(e))
            index += 1
            atesa.update_progress(index / len(filelist))
        results = sorted(results,key=lambda x: abs(x[0]))
        for result in results:
            file.write(result[1] + ': ' + str(result[0]) + '\n')
        file.close()


def committor_analysis(**kwargs):
    """
    Perform committor analysis simulations are write results to committor_analysis.out

    This function makes a new directory in working_directory called committor_analysis to perform its simulations. All
    the parameters it takes are simply passed through from variables of the same names in atesa.py.

    Parameters
    ----------
    candidateops : list
        The parsed list of user-defined candidate OPs.
    rc_definition : str
        The user-defined reaction coordinate equation, containing OP[X] (where X is an integer) for each OP value.
    working_directory : str
        Path to the aimless shooting working directory (contains shooting point coordinate files, as.log, etc.)
    topology : str
        Name of the parameter/topology file for this run.
    rc_minmax : list
        List of format [mins,maxes] where mins and maxes are both lists containing the minimum and maximum observed
        values of each OP, in ascending order, respectively.
    committor_analysis_options : list
        User-defined list of parameters to define committor_analysis options. Format is: [n_shots, rc_ts, rc_tol,
        min_points, min_dist]
        n_shots : int
            Number of simulations per shooting point.
        rc_ts : float
            Value of RC corresponding to the transition state (usually 0).
        rc_tol : float
            Maximum distance from rc_ts for inclusion in the list of shooting points.
        min_points : int
            Number of shooting points to require (difference is made up by performing additional sampling around the TS
            using Equilibrium Path Sampling (see documentation) (Not yet implemented).
        min_dist : int
            Minimum number of shooting moves between two members of the same thread in order to allow them both to be
            included in the list of shooting points to test, to ensure sufficient decorrelation.
    batch_system : str
        Either 'pbs' or 'slurm' to indicate the type of batch system in use.
    commit_define_fwd : list
        User-defined definition of commitment in the "fwd" direction (equivalent to user's commit_fwd input)
    commit_define_bwd : list
        User-defined definition of commitment in the "bwd" direction (equivalent to user's commit_bwd input)
    literal_ops : bool
        Boolean identifying whether candidateops should be interpreted as a list of literal strings to be evaluated
        using eval() (True), or whether the definitions are given using the nested list format.

    Returns
    -------
    None

    """
    try:                                    # import here to avoid circular imports
        from atesa import atesa
    except ImportError:                     # atesa is called as a script instead of installed as a package
        atesa = importlib.import_module('atesa')

    def checkcommit(name, topology, directory=''):
        # Copied directly from atesa.py, with minor edits to make it standalone.

        committor_directory = ''
        if directory:
            directory += '/'
            committor_directory = directory + 'committor_analysis/'

        if not os.path.isfile(committor_directory + name):  # if the file doesn't exist yet, just do nothing
            return ''

        traj = pytraj.iterload(committor_directory + name, directory + topology, format='.nc')

        if not traj:  # catches error if the trajectory file exists but has zero frames
            print(
                'Don\'t worry about this internal error; it just means that atesa is checking for commitment in a trajectory that doesn\'t have any frames yet, probably because the simulation has only just begun.')
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

    # Import necessary variables from atesa.py
    working_directory = kwargs['working_directory']
    rc_definition = kwargs['rc_definition']
    literal_ops = kwargs['literal_ops']
    candidateops = kwargs['candidateops']
    topology = kwargs['topology']
    rc_minmax = kwargs['rc_minmax']
    committor_analysis_options = kwargs['committor_analysis']
    batch_system = kwargs['batch_system']
    commit_define_fwd = kwargs['commit_define_fwd']
    commit_define_bwd = kwargs['commit_define_bwd']
    home_folder = kwargs['home_folder']
    groupfile = kwargs['groupfile']
    env = kwargs['env']
    prod_nodes = kwargs['prod_nodes']
    prod_ppn = kwargs['prod_ppn']
    prod_walltime = kwargs['prod_walltime']
    prod_mem = kwargs['prod_mem']
    init_nodes = kwargs['init_nodes']
    init_ppn = kwargs['init_ppn']
    init_walltime = kwargs['init_walltime']
    init_mem = kwargs['init_mem']
    eps_settings = kwargs['eps_settings']
    include_qdot = kwargs['include_qdot']

    # Import arguments from user's input file
    n_shots = committor_analysis_options[0]
    rc_ts = committor_analysis_options[1]
    rc_tol = committor_analysis_options[2]
    min_points = committor_analysis_options[3]
    min_dist = committor_analysis_options[4]
    committor_suffix = committor_analysis_options[5]

    try:
        os.chdir(working_directory)
    except IOError:
        sys.exit('Error: could not read working directory: ' + working_directory)

    try:
        os.makedirs('committor_analysis' + committor_suffix)
    except FileExistsError:
        pass
    os.chdir('committor_analysis' + committor_suffix)

    try:
        os.remove('ca.log')                 # delete previous run's log file
    except OSError:                         # catches error if no previous log file exists
        pass
    with open('ca.log', 'w+') as newlog:    # make a new log file
        newlog.write('New log file')
        newlog.close()
        logfile = os.getcwd() + '/ca.log'

    # Implementation of committor analysis. Start by obtaining the list of RC values to work with...
    if not os.path.exists(working_directory + '/rc_eval' + committor_suffix + '.out'):
        open(logfile, 'a').write('\nNo rc_eval' + committor_suffix + '.out found in working directory, generating it...')
        return_rcs(**locals())
    else:
        if open(working_directory + '/rc_eval' + committor_suffix + '.out', 'r').readlines():
            open(logfile, 'a').write('\nFound ' + working_directory + '/rc_eval' + committor_suffix + '.out, continuing...')
        else:
            open(logfile, 'a').write('\nFound ' + working_directory + '/rc_eval' + committor_suffix + '.out, but it\'s empty. Generating a new one...')
            return_rcs(**locals())
    open(logfile, 'a').close()

    # A subtlety; rc_eval os.chdir's back to the aimless shooting working directory, so we need to give this line again (maybe)
    os.chdir(working_directory + '/committor_analysis' + committor_suffix)

    eligible = []                               # initialize list of eligible shooting points for committor analysis
    eligible_rcs = []                           # another list holding corresponding rc values
    lines = open('../rc_eval' + committor_suffix + '.out', 'r').readlines()
    open('../rc_eval' + committor_suffix + '.out', 'r').close()
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
        this_kwargs = locals()
        this_kwargs.__delitem__('point')  # so as to avoid passing multiple copies of point to spawnthread()
        thread = atesa.spawnthread(point,type='committor_analysis',suffix='0',**this_kwargs)
        threadlist.append(thread)

    running = []
    for thread in threadlist:
        this_kwargs = locals()
        this_kwargs.__delitem__('thread')   # so as to avoid passing multiple copies of thread to makebatch and subbatch
        this_kwargs.__delitem__('logfile')  # so as to avoid passing multiple copies of logfile to subbatch
        atesa.makebatch(thread,**this_kwargs)
        thread.jobidlist = atesa.subbatch(thread,logfile=logfile,**this_kwargs)
        running.append(thread)
        thread.commitlist = [[] for dummy in range(n_shots)]    # initialize commitlist to contain n_shots values

    # Having submitted the jobs to the batch system, now all I need to do is track their results as in the main code.
    if batch_system == 'pbs':
        cancel_command = 'qdel'
    elif batch_system == 'slurm':
        cancel_command = 'scancel'
    else:
        sys.exit('Error: incorrect batch system type: ' + batch_system)

    # Initialize the output file along with the data headers.
    with open('committor_analysis.out', 'w') as f:
        f.write('Probability of going to \'fwd\' basin     Number of trajectories that committed')
        f.close()

    while running:
        output = atesa.interact('queue',**this_kwargs)   # this_kwargs initially from the above for-loop
        index = 0  # set place in running to 0
        while index < len(running):                 # while instead of for to control indexing manually
            thread = running[index]
            number = -1
            skipcount = 0
            this_kwargs = locals()                        # for passing into interact() and checkcommit()
            this_kwargs.__delitem__('working_directory')  # so as to avoid passing multiple copies
            this_kwargs.__delitem__('thread')
            for jobid in thread.jobidlist:
                number += 1                         # this is the number in range(n_shots) for this jobid
                if not jobid == 'skip' and jobid not in str(output):    # job terminated on its own
                    this_commit = atesa.checkcommit(thread.basename + '_ca_' + str(number) + '.nc',
                                                               thread.prmtop, working_directory, **this_kwargs)
                    if not this_commit:
                        thread.commitlist[number] = 'fail'
                    else:
                        thread.commitlist[number] = this_commit
                    thread.jobidlist[number] = 'skip'                   # PyCharm thinks jobidlist is a str because of the docstring of atesa.subbatch, but rest assured, it is a list
                    open(logfile, 'a').write('\nJob ' + thread.basename + '_ca_' + str(number) + ' has terminated on '
                                             'its own with commitment flag: ' + thread.commitlist[number])
                    open(logfile, 'a').close()
                elif not jobid == 'skip':                               # job is still running
                    this_commit = atesa.checkcommit(thread.basename + '_ca_' + str(number) + '.nc',
                                                               thread.prmtop, working_directory, **this_kwargs)
                    if this_commit:
                        thread.commitlist[number] = this_commit
                        process = subprocess.Popen([cancel_command, thread.jobidlist[number]], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()           # doesn't do anything, I think
                        thread.jobidlist[number] = 'skip'
                        open(logfile, 'a').write('\nJob ' + thread.basename + '_ca_' + str(number) + ' has been '
                                                 'terminated early with commitment flag: ' + thread.commitlist[number])
                        open(logfile, 'a').close()
                elif jobid == 'skip':                                   # job has already been handled
                    skipcount += 1
            if skipcount == n_shots:                # if all the jobs associated with this thread have ended...
                # This block of code meant to provide partial output as committor_analysis runs.
                fwd_count = 0
                bwd_count = 0
                for result in running[index].commitlist:
                    if result == 'fwd':
                        fwd_count += 1
                    elif result == 'bwd':
                        bwd_count += 1
                open(logfile, 'a').write('\nThread ' + thread.basename + ' has finished with ' + str(fwd_count) + ' '
                                         'forward and ' + str(bwd_count) + ' backward trajectories.')
                open(logfile, 'a').close()
                if fwd_count + bwd_count >= 2:  # otherwise we'll always get 1 or 0, never useful
                    pb = fwd_count / (fwd_count + bwd_count)
                    open('committor_analysis.out', 'a').write(str(pb) + ' ' + str(fwd_count+bwd_count) + '\n')  # todo: still writing nothing for some reason. I actually have no evidence that skipcount == n_shots is ever True
                    open('committor_analysis.out', 'a').close()
                del running[index]
                index -= 1
            index += 1

        time.sleep(60)                              # just wait a bit so as not to overwhelm the batch system

    # Finally, we just need to interpret the results and output them in a format that's convenient for a histogram
    # todo: this code currently rechecks every trajectory for commitment, rather than using the values stored in each thread.commitlist. Fix this!
    # trajectories = sorted(glob.glob('*_ca_*.nc'))
    #
    # with open('committor_analysis.out', 'w') as f:
    #     f.write('Probability of going to \'fwd\' basin     Number of trajectories that committed')
    #     f.close()
    #
    # last_basename = ''
    # this_committor = []
    # count = 0
    # for trajectory in trajectories:
    #     count += 1
    #     atesa.update_progress(count / len(trajectories))
    #     basename = trajectory[:-8]  # name excluding '_ca_#.nc' where # is one digit (only works because mine are 0-9)
    #     commit = checkcommit(trajectory, topology, working_directory)  # todo: the offending line for the above todo (other lines may also need to change of course)
    #     if basename != last_basename or trajectory == trajectories[-1]:
    #         if trajectory == trajectories[-1]:  # sloppy, but whatever
    #             this_committor.append(commit)   # do it this way so the last trajectory will also run the below code
    #         # collate results of previous basename
    #         fwd_count = 0
    #         bwd_count = 0
    #         if this_committor:                  # true for every pass except first one
    #             for result in this_committor:
    #                 if result == 'fwd':
    #                     fwd_count += 1
    #                 elif result == 'bwd':
    #                     bwd_count += 1
    #             if fwd_count + bwd_count >= 2:  # otherwise we'll always get 1 or 0, never useful
    #                 pb = fwd_count / (fwd_count + bwd_count)
    #                 open('committor_analysis.out', 'a').write(str(pb) + ' ' + str(fwd_count+bwd_count) + '\n')
    #                 open('committor_analysis.out', 'a').close()
    #         # start new basename
    #         last_basename = basename
    #         this_committor = [commit]
    #     else:                                   # if basename == last_basename and not last trajectory
    #         this_committor.append(commit)

    open(logfile, 'a').write('\nFinished committor analysis on ' + str(len(threadlist)/n_shots) + ' shooting points.'
                              '\nResults in ' + working_directory + '/committor_analysis' + committor_suffix +
                              '/committor_analysis.out')
    open(logfile, 'a').close()
