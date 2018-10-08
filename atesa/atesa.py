#! /usr/bin/env python

# Core Aimless Transition Ensemble Sampling and Analysis (ATESA) code. Tucker Burgin, 2018

from __future__ import division
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
import math
from jinja2 import Environment, FileSystemLoader
#from atesa import rc_eval
rc_eval = importlib.import_module('rc_eval')


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
        self.history = []       # for each job in this thread's history, includes job name and result code: F = both went forward, B = both went backward, S = success, and X = one or both simulations failed
        self.commitlist = []    # for use in committor analysis
        self.jobidlist = []     # for use in committor analysis
        try:
            if eps_settings:
                self.eps_fwd = random.randint(0,eps_settings[1] - 1)    # number of beads to propagate forward in the next step
                self.eps_bwd = eps_settings[1] - self.eps_fwd - 1       # number of beads to propagate backward in the next step
                self.eps_fwd_la = self.eps_fwd                          # value from the last accepted move
                self.eps_bwd_la = self.eps_bwd                          # value from the last accepted move
                self.rc_min = 0     # lower boundary of RC window that this thread started in
                self.rc_max = 0     # upper boundary of RC window that this thread started in
        except NameError:
            pass    # allows Thread() to be called without requiring eps_settings to have been defined yet


# Then, define a series of functions to perform the basic tasks of creating batch files, interacting with the batch
# system, and interpreting/modifying coordinate and trajectory files.
def handle_groupfile(job_to_add=''):
    """
    Update attributes of groupfile_list and return name of the groupfile that should be written to next

    Routine to handle groupfiles when necessary (i.e., when groupfile > 0). Maintains a list object keeping track of
    existing groupfiles and their contents as well as their statuses and ages. This function is called in makebatch()
    to return the name of the groupfile that that function should be writing to, and subbatch() redirects here when
    groupfile > 0. Finally, it is called at the end of every main_loop loop to update statuses as needed.

    job_to_add is an optional argument; if present, this function appends the name of the job given to the <CONTENTS>
    field for the groupfile that it returns. This puts the onus on the code that calls handle_groupfile() to actually
    put that job in that groupfile, of course!

    Process flow for this function:
      1. iterate through existing list of groupfiles as read from groupfile_list
      2. update statuses as necessary
      3. submit any of them that have status "construction" and length == groupfile, or age > groupfile_max_delay and
         length >= 1 (where length means number of lines)
      4. if the list no longer contains any with status "construction", make a new, blank one and return its name
      5. otherwise, return the name of the groupfile with status "construction"

    Format of each sublist of groupfile_list:
      [<NAME>, <STATUS>, <CONTENTS>, <TIME>]
          <NAME> is the name of the groupfile
          <STATUS> is any of:
              "construction", meaning this groupfile is the one to add new lines to (only one of these at a time),
              "completed", meaning the groupfile was submitted and has since terminated,
              "processed", meaning the groupfile has been fed through main_loop and followup jobs submitted, or
              a string of numbers indicating a jobid of a currently-running groupfile
          <CONTENTS> is a list of the jobs contained in the groupfile (their (thread.name + '_' + thread.type) values)
          <TIME> is the time in seconds when the groupfile was created
              if the current time - TIME > groupfile_max_delay, then the groupfile is submitted regardless of length
              (assuming it's at least of length 1)

    Parameters
    ----------
    job_to_add : str
        Name of job to append to the current groupfile's groupfile_list entry. Default = '' (don't add any jobs)

    Returns
    -------
    str
        Name of the groupfile to which new jobs should be appended.

    """

    def new_groupfile(job_to_add_local):
        # Specialized subroutine to build a new empty groupfile and add its information to the groupfile_list
        # If a job_to_add is present, add it to the new groupfile_list entry.
        if groupfile_list:
            pattern = re.compile('\_[0-9]*')
            suffix = pattern.findall(groupfile_list[-1][0])[0][1:]  # this gets the suffix of the last groupfile
            suffix = str(int(suffix) + 1)
        else:
            suffix = '1'
        open('groupfile_' + suffix,'w').close()                     # make a new, empty groupfile
        groupfile_list.append(['groupfile_' + suffix,'construction',job_to_add_local,time.time()])
        return 'groupfile_' + suffix

    def sub_groupfile(groupfile_name):
        # Specialized subroutine to submit a groupfile job as a batch job
        # Creates a batchfile from the template and submits, returning jobid
        batch = 'batch_' + batch_system + '_groupfile.tpl'
        open('as.log', 'a').write('\nWriting groupfile batch file for ' + groupfile_name)
        open('as.log', 'a').close()

        # This block necessary to submit jobs with one fewer than max groups when "flag" from outer scope is True
        if flag:
            groupcount = groupfile - 1
            this_ppn = int(prod_ppn - (prod_ppn/groupfile))  # remove cores proportional to removed simulations
        else:
            groupcount = groupfile
            this_ppn = prod_ppn

        template = env.get_template(batch)
        filled = template.render(nodes=prod_nodes, taskspernode=this_ppn, walltime=prod_walltime, solver='sander',
                                 mem=prod_mem, working_directory=working_directory, groupcount=groupcount,
                                 groupfile=groupfile_name, name=groupfile_name)
        with open(groupfile_name + '_groupfile.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

        output = interact(groupfile_name + '_groupfile.' + batch_system)
        return output


    for groupfile_data in groupfile_list:
        try:                                            # if groupfile was running when this function was last called
            null = int(groupfile_data[1])               # cast to int only works if status is a jobid
            # Check queue to see if it's still running
            output = interact('queue')

            if not groupfile_data[1] in str(output):         # all jobs in the groupfile have finished
                groupfile_data[1] = 'completed'         # update job status
        except ValueError:
            if groupfile_data[1] == 'construction':     # groupfile is the one being built
                # First, check if this groupfile is due to be submitted based on groupfile_max_delay
                age = time.time() - groupfile_data[3]   # age of groupfile entry
                length = len(open(groupfile_data[0],'r').readlines())
                # flag helps us know to submit jobs a little early when there are only prod jobs in the itinerary and
                # no room for both halves of another prod job in the groupfile todo: won't work with committor analysis
                flag = (length == groupfile - 1 and 'init' not in [thread.type for thread in itinerary])
                if length >= 1 and ((age > groupfile_max_delay > 0) or length == groupfile or flag):
                    groupfile_data[1] = sub_groupfile(groupfile_data[0])    # submit groupfile with call to sub_groupfile()
                    return new_groupfile(job_to_add)    # make a new groupfile to return
                else:
                    if job_to_add:                      # add the job_to_add to the contents field, if applicable
                        groupfile_data[2] += job_to_add + ' '
                    return groupfile_data[0]            # this is the current groupfile and it's not full, so return it

    if not groupfile_list:                          # this is the first call to this function, so no groupfiles exist yet
        return new_groupfile(job_to_add)            # make a new groupfile to return


def interact(type,**kwargs):
    """
    Handle submitting of jobs to the batch system or looking up of currently queued and running jobs

    Parameters
    ----------
    type : str
        Two valid options:
        - "queue", to get currently queued and running jobs, or
        - anything else, to submit the batch file given by type to the batch system

    Returns
    -------
    str
        output from batch system depending on the input:
        - "queue", returns raw output
        - anything else, returns the jobid number of the newly submitted batch job (as a string)

    """
    # todo: replace system-specific error handling here with more general solution
    # todo: test this function
    global batch_system

    if kwargs:
        batch_system = kwargs['batch_system']

    user_alias = '$USER'
    command = ''        # line just here to suppress "variable may be called before being defined" error in my IDE
    if type == 'queue':
        if batch_system == 'pbs':
            command = 'qselect -u ' + user_alias + ' -s QR'
        elif batch_system == 'slurm':
            command = 'squeue -u ' + user_alias
    else:               # submitting the batch file given by "type"
        if batch_system == 'pbs':
            command = 'qsub ' + str(type)
        elif batch_system == 'slurm':
            command = 'sbatch ' + str(type)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               close_fds=True, shell=True)
    output = process.stdout.read()
    while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read()
    if 'Bad UID for job execution MSG=user does not exist in server password file' in str(output) or\
       'This stream has already been closed. End of File.' in str(output):
        open('as.log', 'a').write('\nWarning: attempted to submit a job, but got error: ' + str(output) + '\n'
                                  + 'Retrying in one minute...')
        open('as.log', 'a').close()
        time.sleep(60)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read()

    if type == 'queue':     # return raw output
        return output
    else:                   # return jobid of submitted job, or exit with error message if submission failed
        pattern = re.compile('[0-9]+')
        try:
            return pattern.findall(str(output))[0]
        except IndexError:
            sys.exit('Error: unable to submit a batch job: ' + str(type) + '. Got message: ' + str(output))


def makebatch(thread,**kwargs):
    """
    Make the appropriate batch file for the given thread; or, append the appropriate line to the current groupfile

    Has two mutually exclusive branches:
    - When groupfile == 0, makes the batch file appropriate for thread.type, using other thread attributes as necessary,
      by filling a Jinja2 template from the "templates" directory
    - When groupfile > 0, appends a new line to the groupfile indicated by handle_groupfile() appropriate for
      thread.type.

    If thread.type == 'committor_analysis', then the additional parameter n_shots is required to identify the number of
    committor analysis batch jobs to prepare.

    Parameters
    ----------
    thread : Thread
        The Thread object for which to prepare the requisite file(s).
    n_shots : int
        The number of committor analysis shots to make for each shooting point. Default = 0

    Returns
    -------
    None

    """
    global working_directory    # necessary to define these as globals because the if kwargs block below will cause them
    global home_folder          # to be interpreted as locals otherwise, regardless of whether or not kwargs is supplied
    global groupfile
    global env
    global batch_system
    global prod_nodes
    global prod_ppn
    global prod_walltime
    global prod_mem
    global init_nodes
    global init_ppn
    global init_walltime
    global init_mem
    global n_shots
    global eps_settings
    global committor_analysis

    if kwargs:
        working_directory = kwargs['working_directory']
        home_folder = kwargs['home_folder']
        groupfile = kwargs['groupfile']
        env = kwargs['env']
        batch_system = kwargs['batch_system']
        prod_nodes = kwargs['prod_nodes']
        prod_ppn = kwargs['prod_ppn']
        prod_walltime = kwargs['prod_walltime']
        prod_mem = kwargs['prod_mem']
        init_nodes = kwargs['init_nodes']
        init_ppn = kwargs['init_ppn']
        init_walltime = kwargs['init_walltime']
        init_mem = kwargs['init_mem']
        n_shots = kwargs['n_shots']
        eps_settings = kwargs['eps_settings']
        committor_analysis = kwargs['committor_analysis_options']

    name = thread.name
    type = thread.type
    batch = 'batch_' + batch_system + '.tpl'

    if not os.path.exists(home_folder + '/' + 'input_files'):
        sys.exit('Error: could not locate input_files folder: ' + home_folder + '/' + 'input_files\nSee documentation '
                                                                                      'for the \'home_folder\' option.')

    if eps_settings:
        prod_fwd = working_directory + '/eps' + str(thread.eps_fwd) + '.in'
        prod_bwd = working_directory + '/eps' + str(thread.eps_bwd) + '.in'
    else:
        prod_fwd = home_folder + '/input_files/prod.in'
        prod_bwd = home_folder + '/input_files/prod.in'


    # Append a new line to the current groupfile (this is spiritually similar to building a batch file for jobs that are
    # submitted individually; the actual batch file to submit the group file is built by handle_groupfile() when ready)
    # This implementation is ugly because groupfile support was added later in development. If I ever get the chance to
    # go back through and rewrite this code, this is an opportunity for polish!
    if groupfile > 0:
        current_groupfile = handle_groupfile(thread.name + '_' + thread.type)
        if type == 'init':
            inp = home_folder + '/input_files/init.in'
            out = name + '_init.out'
            top = thread.prmtop
            inpcrd = thread.start_name
            rst = name + '_init_fwd.rst'
            nc = name + '_init.nc'
            open(current_groupfile, 'a').write('-i ' + inp + ' -o ' + out + ' -p ' + top + ' -c ' + inpcrd + ' -r ' + rst + ' -x ' + nc + '\n')
            open(current_groupfile, 'a').close()
        elif type == 'prod':
            inp = prod_fwd
            out = name + '_fwd.out'
            top = thread.prmtop
            inpcrd = name + '_init_fwd.rst'
            rst = name + '_fwd.rst'
            nc = name + '_fwd.nc'
            open(current_groupfile, 'a').write('-i ' + inp + ' -o ' + out + ' -p ' + top + ' -c ' + inpcrd + ' -r ' + rst + ' -x ' + nc + '\n')
            # current_groupfile = handle_groupfile() # uncomment this line to enable splitting the halves of prod jobs up among different groupfiles (remember to comment out the "flag" line in handle_groupfile too)
            inp = prod_bwd
            out = name + '_bwd.out'
            top = thread.prmtop
            inpcrd = name + '_init_bwd.rst'
            rst = name + '_bwd.rst'
            nc = name + '_bwd.nc'
            open(current_groupfile, 'a').write('-i ' + inp + ' -o ' + out + ' -p ' + top + ' -c ' + inpcrd + ' -r ' + rst + ' -x ' + nc + '\n')
            open(current_groupfile, 'a').close()
        elif type == 'committor_analysis':
            name = thread.basename
            for i in range(n_shots):
                inp = home_folder + '/input_files/committor_analysis.in'
                out = name + '_ca_' + str(i) + '.out'
                top = '../' + thread.prmtop
                inpcrd = '../' + name
                rst = name + '_ca_' + str(i) + '.rst'
                nc = name + '_ca_' + str(i) + '.nc'
                open(current_groupfile, 'a').write('-i ' + inp + ' -o ' + out + ' -p ' + top + ' -c ' + inpcrd + ' -r ' + rst + ' -x ' + nc + '\n')
                open(current_groupfile, 'a').close()
                # Since we're adding multiple lines here, we need to check at every iteration if a new groupfile is
                # required. handle_groupfile() handles this for us neatly.
                if len(open(current_groupfile, 'r').readlines()) == groupfile:
                    current_groupfile = handle_groupfile(name + '_ca_' + str(i))
        return ''   # do not proceed with the remainder of this function

    if type == 'init':
        # init batch file
        open('as.log', 'a').write('\nWriting init batch file for ' + name + ' starting from ' + thread.start_name)
        open('as.log', 'a').close()
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
        open('as.log', 'a').close()
        template = env.get_template(batch)
        filled = template.render(name=name + '_fwd', nodes=prod_nodes, taskspernode=prod_ppn, walltime=prod_walltime,
                                 solver='sander', inp=prod_fwd, out=name + '_fwd.out', prmtop=thread.prmtop,
                                 inpcrd=name + '_init_fwd.rst', rst=name + '_fwd.rst', nc=name + '_fwd.nc',
                                 mem=prod_mem, working_directory=working_directory)
        with open(name + '_fwd.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

        open('as.log', 'a').write('\nWriting backward batch file for ' + name)
        open('as.log', 'a').close()
        template = env.get_template(batch)
        filled = template.render(name=name + '_bwd', nodes=prod_nodes, taskspernode=prod_ppn, walltime=prod_walltime,
                                 solver='sander', inp=prod_bwd, out=name + '_bwd.out', prmtop=thread.prmtop,
                                 inpcrd=name + '_init_bwd.rst', rst=name + '_bwd.rst', nc=name + '_bwd.nc',
                                 mem=prod_mem, working_directory=working_directory)
        with open(name + '_bwd.' + batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

    elif type == 'committor_analysis':
        name = thread.basename
        for i in range(n_shots):
            template = env.get_template(batch)
            filled = template.render(name=name + '_ca_' + str(i), nodes=prod_nodes, taskspernode=prod_ppn,
                                     walltime=prod_walltime, solver='sander',
                                     inp=home_folder + '/input_files/committor_analysis.in',
                                     out=name + '_ca_' + str(i) + '.out', prmtop='../' + thread.prmtop,
                                     inpcrd='../' + name, rst=name + '_ca_' + str(i) + '.rst',
                                     nc=name + '_ca_' + str(i) + '.nc', mem=prod_mem,
                                     working_directory=working_directory + '/committor_analysis' + committor_analysis[5])
            with open(name + '_ca_' + str(i) + '.' + batch_system, 'w') as newfile:
                newfile.write(filled)
                newfile.close()

    # This error is obsolete, as unless the code is broken in some way it should be unreachable (job type is not a user-
    # supplied value)
    else:
        open('as.log', 'a').write('\nAn incorrect job type \"' + type + '\" was passed to makebatch.')
        open('as.log', 'a').close()
        sys.exit('An incorrect job type \"' + type + '\" was passed to makebatch.')


def subbatch(thread,direction = '',logfile='as.log',**kwargs):
    """
    Submit the appropriate batch file for the given thread and direction to the batch system

    Has two mutually exclusive branches:
    - When groupfile == 0, simply constructs the name of the batch file to submit and calls interact() to do so
    - When groupfile > 0, does nothing (as in this case, batch file submission is handled in handle_groupfile())

    If thread.type == 'committor_analysis', then the additional parameter n_shots is required to identify the number of
    committor analysis batch jobs to submit.

    Parameters
    ----------
    thread : Thread
        The Thread object for which to submit the requisite file(s).
    direction : str
        Either 'fwd', 'bwd', or 'init', as corresponding to the desired simulation type. This option is omitted for
        committor analysis. Default = ''
    n_shots : int
        The number of committor analysis shots to make for each shooting point. Default = 0
    logfile : str
        Name of the logfile to write output to. This is used to redirect the output from the default logfile to ca.log
        when subbatch is called from rc_eval.committor_analysis(). Default = as.log

    Returns
    -------
    str
        if groupfile > 0, returns 'groupfile';
        elif direction == 'fwd', 'bwd', or 'init', directly passes back the output from interact(); else,
    list
        A list object containing n_shots jobid strings corresponding to the jobids of the submitted committor analysis
        simulations.

    """
    name = thread.name  # just shorthand

    global groupfile        # necessary to define these as globals because the if kwargs block below will cause them
    global batch_system     # to be interpreted as locals otherwise, regardless of whether or not kwargs is supplied
    global n_shots

    if kwargs:
        groupfile = kwargs['groupfile']
        batch_system = kwargs['batch_system']
        n_shots = kwargs['n_shots']
    else:
        kwargs = {}

    if groupfile > 0:
        # todo: add support for groupfiles during committor analysis
        # Usually subbatch() returns a jobid, but that's not appropriate when groupfiles are used, as they submit when
        # they're full rather than when subbatch is called. If we're using groupfiles, this function is not needed, and
        # so we simply issue a return statement to break out of it without doing anything.
        return 'groupfile'
    if direction == 'fwd':
        type = 'fwd'
    elif direction == 'bwd':
        type = 'bwd'
    elif direction == 'init':
        type = 'init'
    else:
        type = thread.type
        name = thread.basename

    if type == 'committor_analysis':
        jobids = []
        for i in range(n_shots):
            open(logfile, 'a').write('\nSubmitting job: ' + name + '_ca_' + str(i) + '.' + batch_system)
            open(logfile, 'a').close()
            output = interact(name + '_ca_' + str(i) + '.' + batch_system,**kwargs)
            jobids.append(output)
        return jobids
    else:
        open(logfile, 'a').write('\nSubmitting job: ' + name + '_' + type + '.' + batch_system)
        open(logfile, 'a').close()
        output = interact(name + '_' + type + '.' + batch_system,**kwargs)
        open(logfile, 'a').write('\nBatch system says: ' + str(output)) # todo: put this inside interact? As written, this always prints just the jobid of the submitted job rather than the full message.
        open(logfile, 'a').close()
        return output


def spawnthread(basename, thread_type='init', suffix='',**kwargs):
    """
    Generate a new Thread object with the desired specifications

    This is merely a shortcut function and may be deprecated in future releases in favor of defining Thread() in such a
    way as to make it obsolete. It ensures that Thread.name = Thread.basename + '_' + Thread.suffix.

    Parameters
    ----------
    basename : str
        The initial string shared by all files corresponding to this thread.
    thread_type : str
        Either 'fwd', 'bwd', 'init', or 'committor_analysis', to indicate the type of the next job. Default = 'init'
    suffix : str
        The string to append to the first member of this new Thread. Should be a string containing an integer, as in
        suffix='1'. Default = ''
    prmtop : str
        Name of the topology file to use for this thread. If this option is left unspecified, the value of the input
        file argument "topology" is used. This option is given to allow spawnthread() to be called from outside
        atesa.py without error.

    Returns
    -------
    Thread
        The newly created Thread object.

    """
    global topology         # necessary to define these as globals because the if kwargs block below will cause them to
    global eps_settings     # be interpreted as locals otherwise, regardless of whether or not kwargs is supplied
    global candidateops
    global eps_dynamic_seed

    if kwargs:
        topology = kwargs['topology']
        eps_settings = kwargs['eps_settings']
        candidateops = kwargs['candidateops']

    new_thread = Thread()
    new_thread.basename = basename
    new_thread.suffix = suffix
    new_thread.name = basename + '_' + suffix
    new_thread.type = thread_type
    new_thread.start_name = basename
    new_thread.prmtop = topology

    if eps_settings:    # need to evaluate the window this Thread started in and store its boundaries.
        op_values = [float(op) for op in candidatevalues(basename,reduce=True).split(' ') if op]  # OP values as a list
        equation = rc_definition
        if literal_ops:
            local_candidateops = [candidateops]                 # to fix error where candidateops has unexpected format
        else:
            local_candidateops = candidateops
        if include_qdot:
            qdot_factor = 2  # to include qdot OPs if applicable
        else:
            qdot_factor = 1
        for j in reversed(range(int(qdot_factor * len(local_candidateops[0])))):  # for each candidate op... (reversed to avoid e.g. 'OP10' -> 'op_values[0]0')
            equation = equation.replace('OP' + str(j + 1), 'op_values['+str(j)+']')
        try:
            rc_value = eval(str(equation))
        except TypeError:
            print(str(op_values) + '\n')
            sys.exit(equation)

        if eps_dynamic_seed:
            if not empty_windows:   # if empty_windows is empty, i.e., this is the first thread being spawned
                if type(eps_dynamic_seed) == int:
                    eps_dynamic_seed = [eps_dynamic_seed for null in range(len(eps_windows) - 1)]
                elif not len(eps_dynamic_seed) == (len(eps_windows) - 1):
                    sys.exit('Error: eps_dynamic_seed was given as a list, but is not of the same length as the '
                             'number of EPS windows. There are ' + str((len(eps_windows) - 1)) + ' EPS windows but'
                             ' eps_dynamic_seed is of length ' + str(len(eps_dynamic_seed)))
                window_index = 0
                for window in range(len(eps_windows) - 1):
                    empty_windows.append(eps_dynamic_seed[window_index])  # meaning eps_dynamic_seed[window_index] more threads need to start here before it will no longer be considered "empty"
                    window_index += 1

        for window in range(len(eps_windows)-1):
            # use of inclusive inequality on both sides prevents values equal to the min or max from presenting issues
            if eps_windows[window] - overlap <= rc_value <= eps_windows[window+1] + overlap:
                open('as.log', 'a').write('\nCreating new thread starting from initial structure' + basename + ' with'
                                          ' RC value ' + str(rc_value))
                new_thread.rc_min = eps_windows[window] - overlap
                new_thread.rc_max = eps_windows[window+1] + overlap
                if eps_dynamic_seed:
                    empty_windows[window] -= 1   # set empty_windows for this window to 1 less
                    if empty_windows[window] < 0:
                        empty_windows[window] = 0
                break
        if not new_thread.rc_min and not new_thread.rc_max:     # since if I check just one, it could be zero!
            sys.exit('Error: initial structure ' + basename + ' has RC value (' + str(rc_value) + ') outside the '
                     'defined range (' + str(rc_min) + ' to ' + str(rc_max) + ').')



    return new_thread


def checkcommit(thread,direction,directory='',**kwargs):
    """
    Check a currently running Thread for commitment to either of the user-defined basins and return a string indicating
    the direction of that commitment, if applicable.

    This function loads the indicated trajectory and measures the distances indicated in the commit_fwd and commit_bwd
    lists to compare them against the designated cutoff values. If the user provided basin definitions that are not
    mutually exclusive for some reason (this is strongly cautioned against) and both conditions are met, then this
    function will identify commitment in the 'fwd' direction only.

    Parameters
    ----------
    thread : Thread
        The Thread object whose most recent simulation is to be checked for commitment.
    direction : str
        Either 'fwd', 'bwd'  or 'committor_analysis', to indicate which simulation to check commitment for. Note that
        this parameter only refers to the name of the simulation (production simulations are named with 'fwd' or 'bwd'
        to indicate the original and reversed velocities), not the value that this function returns.
    directory : str
        Path to the directory just above committor_analysis. When this parameter is not provided, the function looks for
        trajectories in the working directory, so this option is required for committor analysis. Default = ''

    Returns
    -------
    str
        Either 'fwd' when the commit_fwd criteria are met; 'bwd' when the commit_bwd criteria are met; or '' when
        neither are met. Alternatively, 'eps' if eps_settings was provided (no computations are performed in this case)

    """
    global commit_define_bwd  # necessary to define these as globals because the if kwargs block below will cause them
    global commit_define_fwd  # to be interpreted as locals otherwise, regardless of whether or not kwargs is supplied
    global candidateops
    global committor_analysis
    global topology

    if kwargs:
        commit_define_fwd = kwargs['commit_define_fwd']
        commit_define_bwd = kwargs['commit_define_bwd']
        committor_analysis = kwargs['committor_analysis_options']
        topology = kwargs['topology']

    if direction not in ['fwd','bwd','init']:  # occurs when this is called by rc_eval.py
        name = thread
    else:
        name = thread.name

    committor_directory = ''            # empty if directory isn't given, so we're looking in working_directory
    if directory:
        directory += '/'
        committor_directory = directory + 'committor_analysis' + committor_analysis[5] + '/'

    if not os.path.isfile(committor_directory + name + '_' + direction + '.nc'):   # if the file doesn't exist yet, just do nothing
        return ''

    traj = pytraj.iterload(committor_directory + name + '_' + direction + '.nc', directory + topology, format='.nc')    # todo: is this used? Should it be moved?

    if not traj:                        # catches error if the trajectory file exists but has zero frames
        print('Don\'t worry about this internal error; it just means that ATESA is checking for commitment '
              'in a trajectory that doesn\'t have any frames yet, probably because the simulation has only just begun.')
        return ''

    commit_flag = ''                    # initialize flag for commitment; this is the value to be returned eventually
    # todo: test this in a standalone context
    # If we're doing EPS, the commitment criterion is that any of the beads reside inside the given range of RC values
    if eps_settings:
        if direction == 'init':
            filename = thread.name + '_init_fwd.rst'
            traj = pytraj.iterload(filename, directory + topology, format='.rst7')
        elif direction in ['fwd','bwd']:
            filename = name + '_' + direction + '.nc'
            traj = pytraj.iterload(filename, directory + topology, format='.nc')
        else:
            sys.exit('Error: checkcommit() was passed an incorrect direction argument: ' + str(direction))
        rc_values = []
        for i in range(traj.__len__()):                         # iterate through frames
            op_values = [float(op) for op in candidatevalues(filename, frame=i, reduce=True).split(' ') if op]  # OP values as a list
            equation = rc_definition
            if literal_ops:
                local_candidateops = [candidateops]             # to fix error where candidateops has unexpected format
            else:
                local_candidateops = candidateops
            if include_qdot:
                qdot_factor = 2  # to include qdot OPs if applicable
            else:
                qdot_factor = 1
            for j in reversed(range(int(qdot_factor * len(local_candidateops[0])))):     # for each candidate op... (reversed to avoid e.g. 'OP10' -> 'op_values[0]0')
                equation = equation.replace('OP' + str(j + 1), 'op_values['+str(j)+']')
            rc_values.append(eval(str(equation)))
        commit_flag = 'False'           # return 'False' if no beads are in bounds
        for value in rc_values:
            if thread.rc_min <= value <= thread.rc_max:
                commit_flag = 'True'    # return 'True' if any of the beads are in bounds.
                break
    else:
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
                open('as.log', 'a').close()
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
                    open('as.log', 'a').close()
                    sys.exit('An incorrect commitor definition \"' + commit_define_bwd[3][i] + '\" was given for index ' + str(i) +' in the \'bwd\' direction.')

    del traj    # to ensure cleanup of iterload object

    return commit_flag


def pickframe(thread, direction, forked_from=Thread(), frame=-1, suffix=1):
    """
    Write a new restart-format coordinate file using a randomly chosen frame with index between 0 and n_adjust to serve
    as the initial coordinates for a new simulation.

    This function loads the last valid trajectory from thread in the given direction in order to select a random frame.
    If forked_from is provided, it indicates that thread should be treated as a new Thread object but that its initial
    coordinates should be chosen from the forked_from Thread.

    If a frame argument >= 0 is given, this function does not use n_adjust; instead, it simply uses that frame.

    Parameters
    ----------
    thread : Thread
        The Thread object whose most recent simulation is to be checked for commitment.
    direction : str
        Either 'fwd' or 'bwd', to indicate which simulation to pick the random frame from.
    forked_from : Thread
        A Thread object (not the same as thread) from which to draw the initial coordinates for thread. Default =
        Thread() (an empty Thread object, which will not be used.)
    frame : int
        Manually override the random frame picking by providing the frame index to use. Must be >= 0 to be used.

    Returns
    -------
    str
        Filename of the newly created restart-format coordinate file.

    """
    if not isinstance(thread,Thread):
        basename = thread   # thread should be a string
        last_valid = 1
    else:
        basename = thread.basename
        last_valid = thread.last_valid
        suffix = thread.suffix

    if last_valid == '0':           # to catch case where 'last_valid' is still the initial shooting point because a new valid transition path has not been found
        return thread.start_name    # will cause the new start_name to be unchanged from the previous run

    if frame >= 0:
        frame_number = frame
    else:
        frame_number = random.randint(0, n_adjust-1)

    if forked_from.basename:    # if an actual thread was given for forked_from (as opposed to the default)...
        if direction in ['fwd','bwd']:
            traj = pytraj.iterload(forked_from.basename + '_' + forked_from.last_valid + '_' + direction + '.nc', forked_from.prmtop, format='.nc',frame_slice=(frame_number,frame_number+1))
        elif direction == 'init':
            traj = pytraj.iterload(forked_from.basename + '_' + forked_from.last_valid + '_init_fwd.rst', forked_from.prmtop, format='.rst7')
        else:
            sys.exit('Error: pickframe() encountered unexpected direction: ' + direction + ' during fork')
        new_suffix = str(suffix)
    else:
        traj = pytraj.iterload(basename + '_' + last_valid + '_' + direction + '.nc', topology, format='.nc',frame_slice=(frame_number,frame_number+1))
        new_suffix = str(int(suffix) + 1)

    new_restart_name = basename + '_' + new_suffix + '.rst7'
    pytraj.write_traj(new_restart_name,traj,options='multi')    # multi because not including this option seems to keep it on anyway, so I want to be consistent
    try:
        os.rename(new_restart_name + '.1',new_restart_name)     # I don't quite know why, but pytraj appends '.1' to the filename, so this removes it.
    except OSError: # I sort of anticipate this breaking down the line, so this block is here to help handle that.
        open('as.log', 'a').write('\nWarning: tried renaming .rst7.1 file ' + new_restart_name + '.1 with .rst7, but '
                                  'encountered OSError exception. Either you ran out of storage space, or this is a '
                                  'possible indication of an unexpected pytraj version?')
        open('as.log', 'a').close()
        if not os.path.exists(new_restart_name):
            open('as.log', 'a').write('Error: pickframe did not produce the restart file for the next shooting move. '
                                      'Please ensure that you didn\'t run out of storage space, and then raise this '
                                      'issue on GitHub to let me know!')
            sys.exit('Error: pickframe did not produce the restart file for the next shooting move. Please ensure that '
                     'you didn\'t run out of storage space, and then raise this issue on GitHub to let me know!')
        else:
            open('as.log', 'a').write('\nWarning: it tentatively looks like this should be okay, as the desired file was still created.')
            open('as.log', 'a').close()
        pass

    if forked_from.basename:
        open('as.log', 'a').write('\nForking ' + basename + ' from ' + forked_from.basename + '_' + forked_from.last_valid + '_' + direction + '.nc, frame number ' + str(frame_number))
    else:
        open('as.log', 'a').write('\nInitializing next shooting point from shooting run ' + basename + '_' + last_valid + ' in ' + direction + ' direction, frame number ' + str(frame_number))
    open('as.log', 'a').close()

    del traj  # to ensure cleanup of iterload object

    return new_restart_name


def cleanthread(thread):
    """
    Reset thread parameters in preparation for the next step of aimless shooting after the previous one has completed.
    Add the next step to the itinerary if appropriate. Also write to history and output files, implement fork if
    necessary, and terminate the thread if any of the termination criteria are met.

    This function should be called after every thread step is completed to handle it in the appropriate manner. In
    effect, it serves as a housekeeping function to take care of all the important details that are checked for after
    every "prod" step.

    Parameters
    ----------
    thread : Thread
        The Thread object that just completed.

    Returns
    -------
    None

    """
    global candidateops

    def report_rc_values(coord_file):
        # Simple function for outputting the RC values for a given trajectory traj to the eps_results.out file
        # todo: replace use of traj with simple evaluation of eps_fwd/bwd variable, depending on direction argument
        rc_values = []
        if '.rst' in coord_file or '.rst7' in coord_file:
            fileformat = '.rst7'
        elif '.nc' in coord_file:
            fileformat = '.nc'
        else:
            sys.exit('Error: cleanthread.report_rc_values() encountered a file of unknown format: ' + coord_file)
        traj = pytraj.iterload(coord_file, thread.prmtop, format=fileformat)
        for i in range(traj.__len__()):                         # iterate through frames of traj
            op_values = [float(op) for op in candidatevalues(coord_file, frame=i, reduce=True).split(' ') if op]  # OP values as a list
            equation = rc_definition
            if literal_ops:
                local_candidateops = [candidateops]                   # to fix error where candidateops has unexpected format
            else:
                local_candidateops = candidateops
            # todo: should probably convert this rc evaluating code into a standalone function, since it appears in three different places in this file
            if include_qdot:
                qdot_factor = 2  # to include qdot OPs if applicable
            else:
                qdot_factor = 1
            for j in reversed(range(int(qdot_factor * len(local_candidateops[0])))):     # for each candidate op... (reversed to avoid e.g. 'OP10' -> 'op_values[0]0')
                equation = equation.replace('OP' + str(j + 1), 'op_values['+str(j)+']')
            rc_values.append(eval(str(equation)))
        for value in rc_values:
            if thread.rc_min <= value <= thread.rc_max:     # only write to output if the bead is inside the window
                open('eps_results.out', 'a').write(str(thread.rc_min) + ' ' + str(thread.rc_max) + ' ' + str(value) + '\n')
                open('eps_results.out', 'a').close()    # todo: should this be as.out?
        return rc_values

    if eps_settings:                # EPS behavior
        if thread.last_valid == thread.suffix:  # if this move was accepted...
            thread.eps_fwd_la = thread.eps_fwd  # update "last accepted" eps_(b/f)wd attributes for this thread
            thread.eps_bwd_la = thread.eps_bwd
        # Store RC values for each frame in both the fwd and bwd trajectories of the last-accepted move, regardless of
        # whether that's this newest one or an old one.
        fwd_rc_values = []
        bwd_rc_values = []
        init_rc_value = []
        if thread.eps_fwd_la > 0 and int(thread.last_valid) > 0:     # latter requirement because we need at least one accepted trajectory before we can start reporting values
            try:
                fwd_rc_values = report_rc_values(thread.basename + '_' + thread.last_valid + '_fwd.nc')
            except ValueError:
                sys.exit('Debug: Failed on ' + thread.basename + '_' + thread.last_valid + '_fwd.nc'
                         + '\n  thread.eps_fwd_la = ' + str(thread.eps_fwd_la)
                         + '\n  thread.last_valid = ' + str(thread.last_valid)
                         + '\n  thread.suffix = ' + str(thread.suffix))
        if thread.eps_bwd_la > 0 and int(thread.last_valid) > 0:
            try:
                bwd_rc_values = report_rc_values(thread.basename + '_' + thread.last_valid + '_bwd.nc')
            except ValueError:
                sys.exit('Debug: Failed on ' + thread.basename + '_' + thread.last_valid + '_fwd.nc'
                         + '\n  thread.eps_fwd_la = ' + str(thread.eps_fwd_la)
                         + '\n  thread.last_valid = ' + str(thread.last_valid)
                         + '\n  thread.suffix = ' + str(thread.suffix))
        if int(thread.last_valid) > 0:
            init_rc_value = report_rc_values(thread.basename + '_' + thread.last_valid + '_init_fwd.rst')
        # Finally, handle dynamic seeding:
        if eps_dynamic_seed and (True in [bool(x) for x in empty_windows]) and (thread.last_valid == thread.suffix): # todo: is this last boolean required? I think maybe yes because I'm using pickframe()?
            rc_values = list(reversed(bwd_rc_values)) + init_rc_value + fwd_rc_values
            start_bead = 0
            suffix = 1
            for rc_value in rc_values:
                start_bead += 1
                for window in range(len(eps_windows) - 1):
                    if (empty_windows[window] > 0) and (eps_windows[window] - overlap <= rc_value <= eps_windows[window + 1] + overlap):
                        # Write a new coordinate file from the appropriate trajectory
                        # todo: this is so ugly because I didn't design pickframe() to help make a new thread with an unknown initial structure. Can I clean this up somehow?
                        if start_bead <= thread.eps_bwd_la:         # use la values since pickframe uses the la trajectory
                            structure = pickframe(thread.name, 'bwd', frame=int(thread.eps_bwd_la - start_bead), forked_from=thread, suffix=suffix)     # "frame" should be zero-indexed
                            suffix += 1
                            debug_dir = 'bwd'
                            debug_frame = int(thread.eps_bwd_la - start_bead)
                        elif start_bead == thread.eps_bwd_la + 1:   # the initial coordinates
                            structure = pickframe(thread.name, 'init', forked_from=thread, suffix=suffix)
                            suffix += 1
                            debug_dir = 'init_fwd'
                            debug_frame = 'N/A'
                        else:                                       # inside the fwd trajectory
                            structure = pickframe(thread.name, 'fwd', frame=int(start_bead - thread.eps_bwd_la - 2), forked_from=thread, suffix=suffix)   # "frame" should be zero-indexed
                            suffix += 1
                            debug_dir = 'fwd'
                            debug_frame = int(start_bead - thread.eps_bwd_la - 1)
                        newthread = spawnthread(structure, suffix='1')  # spawn a new thread with the default settings
                        allthreads.append(newthread)   # add it to the list of all threads for bookkeeping
                        newthread.last_valid = '0'     # so that if the first shooting point does not result in a valid transition path, shooting will begin from the TS guess
                        newthread.prmtop = topology    # set prmtop filename for the thread
                        itinerary.append(newthread)    # submit it to the itinerary
                        open('as.log', 'a').write('\nEmpty EPS window with upper and lower boundaries: ' +
                                                  str(eps_windows[window] - overlap) + ' and ' +
                                                  str(eps_windows[window + 1] + overlap) + ' has been seeded using '
                                                  'bead ' + str(start_bead) + ' from shooting move ' + thread.name +
                                                  '. Debug information:')
                        open('as.log', 'a').write('\n  fwd_rc_values = ' + str(fwd_rc_values))
                        open('as.log', 'a').write('\n  bwd_rc_values = ' + str(bwd_rc_values))
                        open('as.log', 'a').write('\n  rc_values = ' + str(rc_values))
                        open('as.log', 'a').write('\n  start_bead = ' + str(start_bead))
                        open('as.log', 'a').write('\n  pickframe trajectory = ' + thread.basename + '_' + thread.last_valid + '_' + debug_dir + '.nc')
                        open('as.log', 'a').write('\n  frame from trajectory = ' + str(debug_frame))
                        open('as.log', 'a').write('\n  structure = ' + str(structure))
                        open('as.log', 'a').write('\n  new empty_windows = ' + str(empty_windows))
                        open('as.log', 'a').close()

    elif thread.commit1 != 'fail':  # standard aimless shooting behavior
        # Record result of forward trajectory in output file. This is done regardless of whether the shooting point was
        # accepted; accept/reject is for keeping the sampling around the separatrix, but even rejected points are valid
        # for calculating the reaction coordinate so long as they committed to a basin!
        if thread.commit1 == 'fwd':
            basin = 'A'
        elif thread.commit1 == 'bwd':
            basin = 'B'
        else:
            basin = thread.commit1
            sys.exit('Error: thread commit1 flag took on unexpected value: ' + basin + '\nThis is a weird error.'
                     ' Please raise this issue on GitHub!')
        open('as.out', 'a').write(basin + ' <- ' + candidatevalues(thread.name + '_init_fwd.rst') + '\n')
        open('as.out', 'a').close()

    # Write last result to history
    if thread.last_valid == thread.suffix:
        code = 'S'
    elif thread.commit1 == thread.commit2 == 'fwd':
        code = 'F'
    elif thread.commit1 == thread.commit2 == 'bwd':
        code = 'B'
    else:   # failure of one or both halves of shooting move
        code = 'X'
    thread.history.append(thread.name + ' ' + code)
    with open('history/' + thread.basename, 'w') as file:
        for history_line in thread.history:
            file.write(history_line + '\n')
        file.close()

    thread.total_moves += 1
    open('as.log', 'a').write('\nShooting run ' + thread.name + ' finished with fwd trajectory result: ' + thread.commit1 + ' and bwd trajectory result: ' + thread.commit2)
    if eps_settings:
        open('as.log', 'a').write(', as well as init result: ' + checkcommit(thread, 'init'))   # todo: should probably save an init_commit attribute to threads to avoid checking commitment on init for a second time here.
    open('as.log', 'a').write('\n' + thread.basename + ' has a current acceptance ratio of: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or ' + str(100*thread.accept_moves/thread.total_moves)[0:5] + '%')
    open('as.log', 'a').close()

    # Implementation of fork. Makes (fork - 1) new threads from successful runs and adds them to the itinerary. The new
    # threads do not inherit anything from their parents except starting point and history.
    # todo: test fork with EPS
    if fork > 1 and thread.last_valid == thread.suffix:
        for i in range(fork - 1):
            direction = random.randint(0, 1)
            if direction == 0:
                pick_dir = 'fwd'
            else:
                pick_dir = 'bwd'
            newthread = spawnthread(thread.name + '_' + str(i + 1), suffix='1')
            newthread.prmtop = thread.prmtop
            newthread.start_name = pickframe(newthread, pick_dir, thread)
            newthread.last_valid = '0'
            newthread.history = thread.history
            itinerary.append(newthread)
            allthreads.append(newthread)

    if eps_settings:    # EPS behavior
        start_bead = random.randint(1, k_beads)
        # Thread has attributes eps_fwd and eps_bwd telling me how long the fwd and bwd trajectories are...
        if start_bead <= thread.eps_bwd_la:         # use la values since pickframe uses the la trajectory
            thread.start_name = pickframe(thread, 'bwd', frame=int(thread.eps_bwd_la - start_bead))         # "frame" should be zero-indexed
        elif start_bead == thread.eps_bwd_la + 1:   # the initial coordinates
            thread.start_name = thread.name + '_init_fwd.rst'
        else:                                       # inside the fwd trajectory
            thread.start_name = pickframe(thread, 'fwd', frame=int(start_bead - thread.eps_bwd_la - 2))     # "frame" should be zero-indexed
        thread.eps_fwd = k_beads - start_bead           # set new eps_fwd and _bwd to keep string length the same
        thread.eps_bwd = k_beads - thread.eps_fwd - 1   # extra -1 to account for starting point
    else:               # normal aimless shooting behavior
        direction = random.randint(0, 1)  # I'm not sure doing this helps, but I am sure doing it doesn't hurt
        if direction == 0:
            pick_dir = 'fwd'
        else:
            pick_dir = 'bwd'
        if thread.last_valid == thread.suffix or always_new:  # pick a new starting point if the last move was a success
            thread.start_name = pickframe(thread, pick_dir)

    thread.type = 'init'
    thread.suffix = str(int(thread.suffix) + 1)
    thread.name = thread.basename + '_' + thread.suffix
    thread.jobid1 = ''  # required if eps_settings is given
    thread.jobid2 = ''  # required if eps_settings is given
    thread.commit1 = ''
    thread.commit2 = ''

    if thread.failcount >= max_fails > 0:
        thread.status = 'max_fails'     # the thread dies because it has failed too many times in a row
    elif thread.total_moves >= max_moves > 0:
        thread.status = 'max_moves'     # the thread dies because it has performed too many total moves
    elif thread.accept_moves >= max_accept > 0:
        thread.status = 'max_accept'    # the thread dies because it has accepted too many moves
    else:
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
        file.close()


def candidatevalues(coord_file,frame=-1,reduce=False):
    """
    Evaluate the candidate OP values from the initial coordinates of the most recent step of the given thread.

    This function is only capable of returning order parameter rate of change values when the literal_ops Boolean is
    True. This will change in future versions.

    Parameters
    ----------
    coord_file : str
        The name of the coordinate file to evaluate.
    frame : int
        Frame index to evaluate instead of the most recent. Must be >= 0 to be used.
    reduce : bool
        Boolean indicating whether or not to reduce the candidate OP values based on the contents of rc_minmax. This
        should be used when the output of candidatevalues is being used to calculate an RC value.

    Returns
    -------
    str
        A space-separated series of order parameter values in the same order as they are given in candidate_op, followed
        by their rates of change per 1/20.455 ps (unit of time in Amber) in the same order if applicable.

    """
    global topology

    def reduce_op(input,index):
        # Return the reduced value of the index'th OP with un-reduced value input (an int or float).
        try:
            if not rc_minmax[0][index] or not rc_minmax[1][index]:  # if there's a blank entry in rc_minmax
                sys.exit('Error: rc_definition contains reference to OP' + str(index+1) + ' without a corresponding entry in rc_minmax')
        except IndexError:                                          # if there's no entry at all
            sys.exit('Error: rc_definition contains reference to OP' + str(index + 1) + ' without a corresponding entry in rc_minmax')
        return (input - rc_minmax[0][index])/(rc_minmax[1][index] - rc_minmax[0][index])

    if '.rst' in coord_file or '.rst7' in coord_file:
        fileformat = '.rst7'
    elif '.nc' in coord_file:
        fileformat = '.nc'
    else:
        sys.exit('Error: candidatevalues() encountered a file of unknown format: ' + coord_file)

    output = ''
    if frame >= 0:
        traj = pytraj.iterload(coord_file, topology, format=fileformat, frame_slice=(frame,frame+1))
        if include_qdot:
            pytraj.write_traj('temp_frame.rst7',traj,overwrite=True,velocity=True,options='multi')    # multi because not including this option seems to keep it on anyway, so I want to be consistent
            try:
                os.rename('temp_frame.rst7.1','temp_frame.rst7')    # I don't quite know why, but pytraj appends '.1' to the filename, so this removes it.
            except OSError:                                         # I sort of anticipate this breaking down the line, so this block is here to help handle that.
                open('as.log', 'a').write('\nWarning: tried renaming .rst7.1 file temp_frame.rst7.1 with .rst7, but '
                                          'encountered OSError exception. Either you ran out of storage space, or this '
                                          'is a possible indication of an unexpected pytraj version?')
                open('as.log', 'a').close()
                if not os.path.exists('temp_frame.rst7'):
                    sys.exit('Error: pickframe did not produce the restart file for the next shooting move. Please ensure that you didn\'t run out of storage space, and then raise this issue on GitHub to let me know!')
                else:
                    open('as.log', 'a').write('\nWarning: it tentatively looks like this should be okay, as the desired file was still created.')
                    open('as.log', 'a').close()
                pass
    else:
        traj = pytraj.iterload(coord_file, topology, format=fileformat)

    def increment_coords():
        # Modified from revvels() to increment coordinate values by velocities, rather than reversing velocities.
        # Returns the name of the newly-created coordinate file
        if not frame >= 0:
            filename = coord_file
        else:
            filename = 'temp_frame.rst7'
        byline = open(filename).readlines()
        pattern = re.compile('[-0-9.]+')            # regex to match numbers including decimals and negatives
        pattern2 = re.compile('\s[-0-9.]+')         # regex to match numbers including decimals and negatives, with one space in front
        n_atoms = pattern.findall(byline[1])[0]     # number of atoms indicated on second line of .rst file

        shutil.copyfile(filename, 'temp.rst')
        for i, line in enumerate(fileinput.input('temp.rst', inplace=1)):
            if int(n_atoms)/2 + 2 > i >= 2:
                newline = line
                coords = pattern2.findall(newline)                                          # line of coordinates
                try:
                    vels = pattern2.findall(byline[i + int(math.ceil(int(n_atoms)/2))])     # corresponding velocities
                except IndexError:
                    sys.exit('Error: candidatevalues.increment_coords() encountered an IndexError. This is caused '
                             'by attempting to read qdot values from a coordinate file lacking velocity information, or'
                             ' else by that file being truncated. The offending file is: ' + filename)
                for index in range(len(coords)):
                    length = len(coords[index])                     # length of string representing this coordinate
                    replace_string = ' ' + str(float(coords[index]) + float(vels[index]))[0:length-1]
                    while len(replace_string) < length:
                        replace_string += '0'
                    newline = newline.replace(coords[index], replace_string)
                sys.stdout.write(newline)
            else:
                sys.stdout.write(line)

        return 'temp.rst'

    # Implementation of explicit, to directly interpret user-supplied OPs
    if literal_ops:
        values = []
        index = -1
        for op in candidateops:
            index += 1
            evaluation = eval(op)
            if include_qdot:    # want to save values for later
                values.append(float(evaluation))
            if reduce:
                evaluation = reduce_op(evaluation, index)
            output += str(evaluation) + ' '
        if include_qdot:        # if True, then we want to include rate of change for every OP, too
            # Strategy here is to write a new temporary .rst7 file by incrementing all the coordinate values by their
            # corresponding velocity values, load it as a new iterload object, and then rerun our analysis on that.
            traj = pytraj.iterload(increment_coords(), topology)
            index = -1
            for op in candidateops:
                index += 1
                evaluation = eval(op) - values[index]
                if reduce:
                    evaluation = reduce_op(evaluation,index + len(candidateops))
                output += str(evaluation) + ' '     # Subtract value of op from value 1/20.455 ps earlier

    # todo: implement include_qdot for this branch of candidatevalues, too, or else remove the option to give candidateops this way?
    else:
        for index in range(0,len(candidateops[0])):
            if len(candidateops) == 4:          # candidateops contains dihedrals
                if candidateops[3][index]:      # if this OP is a dihedral
                    value = pytraj.dihedral(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' + candidateops[2][index] + ' ' + candidateops[3][index])[0]
                elif candidateops[2][index]:    # if this OP is an angle
                    value = pytraj.angle(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' + candidateops[2][index])[0]
                else:                           # if this OP is a distance
                    value = pytraj.distance(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index])[0]
            elif len(candidateops) == 3:        # candidateops contains angles but not dihedrals
                if candidateops[2][index]:      # if this OP is an angle
                    value = pytraj.angle(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index] + ' ' + candidateops[2][index])[0]
                else:                           # if this OP is a distance
                    value = pytraj.distance(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index])[0]
            else:                               # candidateops contains only distances
                value = pytraj.distance(traj,mask=candidateops[0][index] + ' ' + candidateops[1][index])[0]

            if reduce:
                value = reduce_op(value, index)
            output += str(value) + ' '

    del traj  # to ensure cleanup of iterload object

    return output


def revvels(thread):
    """
    Write a new restart-format coordinate file by multiplying the velocity values from the most recent "*_init_fwd.rst"
    file belonging to thread by -1.

    This function provides the initial coordinates for "bwd" production simulations in ATESA. Velocities in
    .rst files are stored in groups of three just like coordinates and directly following them. This function finds the
    place where coordinates end and velocities begin by reading the number of atoms listed on the 2nd line of the .rst
    file, dividing it by two, and then navigating to the line immediately after that plus three (since coordinates begin
    on line 3) and minus one (because of indexing).

    Parameters
    ----------
    thread : Thread
        The Thread object that owns the *_init_fwd.rst file to reverse.

    Returns
    -------
    None

    """
    byline = open(thread.name + '_init_fwd.rst').readlines()
    open(thread.name + '_init_fwd.rst').close()
    pattern = re.compile('[-0-9.]+')        # regex to match numbers including decimals and negatives
    pattern2 = re.compile('\s[-0-9.]+')     # regex to match numbers including decimals and negatives, with one space in front
    n_atoms = pattern.findall(byline[1])[0] # number of atoms indicated on second line of .rst file
    offset = 2                              # appropriate for n_atoms is odd; offset helps avoid modifying the box line
    if int(n_atoms) % 2 == 0:               # if n_atoms is even...
        offset = 1                          # appropriate for n_atoms is even

    shutil.copyfile(thread.name + '_init_fwd.rst', thread.name + '_init_bwd.rst')
    for i, line in enumerate(fileinput.input(thread.name + '_init_bwd.rst', inplace=1)):
        if int(n_atoms)/2 + 2 <= i <= int(n_atoms) + offset:      # if this line is a velocity line
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
    """
    Perform the sequence of simulations constituting a full ATESA run.

    This is the primary runtime loop of this program, from which all the helper functions above are called (either
    directly or by one another). Along with the helper functions, it builds Threads, submits them to the batch system,
    monitors them for completion, and handles the output and next steps once they've finished.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # These bash commands will be used later to cancel jobs after they've completed
    if batch_system == 'pbs':
        cancel_command = 'qdel'
    elif batch_system == 'slurm':
        cancel_command = 'scancel'

    global itinerary                # get global variables from outer scope so they are accessible by main_loop()
    global running                  # honestly I don't know why this is necessary, but removing these lines breaks it
    global allthreads

    while itinerary or running:     # while either list has contents...
        itin_names = [thread.name + '_' + thread.type for thread in itinerary]
        run_names = [thread.name + '_' + thread.type for thread in running]
        open('as.log', 'a').write(
            '\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))
        open('as.log', 'a').write('\nSubmitting jobs in itinerary...')
        open('as.log', 'a').close()

        index = -1  # set place in itinerary to -1
        for thread in itinerary:    # for each thread that's ready for its next step...
            index += 1              # increment place in itinerary
            makebatch(thread)       # make the necessary batch file
            if thread.type == 'init':
                thread.jobid1 = subbatch(thread, 'init')    # submit that batch file and collect its jobID into thread
            else:
                try:
                    if not thread.eps_fwd == 0:     # if this setting exists and isn't zero
                        thread.jobid1 = subbatch(thread, 'fwd')
                    else:
                        thread.commit1 = 'False'    # simulation doesn't run, so it doesn't satisfy acceptance criterion
                except NameError:                   # if this setting doesn't exist (normal aimless shooting behavior)
                    thread.jobid1 = subbatch(thread, 'fwd')
                try:                                # repeat for bwd half of thread
                    if not thread.eps_bwd == 0:
                        thread.jobid2 = subbatch(thread, 'bwd')
                    else:
                        thread.commit2 = 'False'
                except NameError:
                    thread.jobid2 = subbatch(thread, 'bwd')
            running.append(itinerary[index])                # move element of itinerary into next position in running

        itinerary = []  # empty itinerary

        itin_names = [thread.name + '_' + thread.type for thread in itinerary]
        run_names = [thread.name + '_' + thread.type for thread in running]
        open('as.log', 'a').write('\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))
        open('as.log', 'a').close()

        # Once again, the fact that groupfile support was only added late in development means that this is a bit ugly.
        # This while loop serves the same purpose as the one directly following it, but works in terms of groupfile jobs
        # instead of individual batch jobs. I can't think of an easy way to unify them, but this is an opportunity for
        # polish if I ever rewrite this software from the ground up.
        #
        # I'm not implementing flexible-length shooting for groupfiles, since it should be exceedingly rare anyway for
        # every job in a groupfile to be committed substantially before the job is ending anyway, assuming the user
        # chooses prod_walltime judiciously.
        # todo: implement EPS for groupfile > 0
        while groupfile > 0 and not itinerary:
            handle_groupfile()                          # update groupfile_list
            for groupfile_data in groupfile_list:       # for each list element [<NAME>, <STATUS>, <CONTENTS>, <TIME>]
                if groupfile_data[1] == 'completed':    # if this groupfile task has status "completed"
                    for string in groupfile_data[2].split(' '):     # iterate through thread.name + '_' + thread.type of jobs in groupfile
                        if 'init' in string:                        # do this for all init jobs in this groupfile
                            # Need to get ahold of the thread this init job represents...
                            for thread in running:       # can't say this is surely the best way to do this...
                                if thread.name in string:
                                    try:
                                        revvels(thread)     # make the initial .rst file for the bwd trajectory
                                        itinerary.append(thread)
                                        thread.type = 'prod'
                                        open('as.log', 'a').write('\nJob completed: ' + thread.name + '_init\nAdding ' + thread.name + ' forward and backward jobs to itinerary')
                                        open('as.log', 'a').close()
                                    except IOError:   # when revvels can't find the init .rst file
                                        open('as.log', 'a').write('\nThread ' + thread.basename + ' crashed: initialization did not produce a restart file.')
                                        if not restart_on_crash:
                                            open('as.log', 'a').write('\nrestart_on_crash = False; thread will not restart')
                                            open('as.log', 'a').close()
                                        elif not restart_on_crash:
                                            open('as.log', 'a').write('\nrestart_on_crash = True; resubmitting thread to itinerary')
                                            open('as.log', 'a').close()
                                            itinerary.append(thread)
                                            thread.type = 'init'
                                    running.remove(thread)
                        if 'prod' in string:                        # do this for all prod jobs in this groupfile
                            # todo: figure out how to tolerate only one of the halves of a prod job showing up in a given groupfile
                            # Need to get ahold of the thread this prod job represents...
                            for thread in running:       # can't say this is surely the best way to do this...
                                if thread.name in string:
                                    thread.commit1 = checkcommit(thread, 'fwd')  # check for commitment in fwd job
                                    if not thread.commit1:
                                        thread.commit1 = 'fail'
                                    thread.commit2 = checkcommit(thread, 'bwd')  # check for commitment in bwd job
                                    if not thread.commit2:
                                        thread.commit2 = 'fail'
                                    running.remove(thread)
                                    thread.failcount += 1       # increment fails in a row regardless of outcome
                                    if eps_settings and thread.commit1 == 'True' or thread.commit2 == 'True':
                                        thread.last_valid = thread.suffix
                                        thread.accept_moves += 1
                                        thread.failcount = 0    # reset fail count to zero if this move was accepted
                                    elif thread.commit1 != thread.commit2 and thread.commit1 != 'fail' and thread.commit2 != 'fail':  # valid transition path, update 'last_valid' attribute
                                        thread.last_valid = thread.suffix
                                        thread.accept_moves += 1
                                        thread.failcount = 0    # reset fail count to zero if this move was accepted
                                    cleanthread(thread)
                    groupfile_data[1] = 'processed'     # to indicate that this completed groupfile job has been handled

            pickle.dump(allthreads, open('restart.pkl', 'wb'))  # dump information required to restart ATESA
            if not itinerary:
                time.sleep(60)                          # delay 60 seconds before checking for job status again

        while groupfile == 0 and not itinerary:     # while itinerary is empty and we're not in groupfile mode...
            output = interact('queue')              # retrieves string containing jobids of running and queued jobs
            index = 0  # set place in running to 0
            while index < len(running):  # while instead of for to control indexing manually
                thread = running[index]
                if thread.jobid1 and not thread.jobid1 in str(output):  # if a submitted job is no longer running
                    if thread.type == 'init':
                        try:
                            revvels(thread)  # make the initial .rst file for the bwd trajectory
                            itinerary.append(running[index])
                            thread.type = 'prod'
                            thread.type = 'prod'
                            open('as.log', 'a').write(
                                '\nJob completed: ' + thread.name + '_init\nAdding ' + thread.name + ' forward and backward jobs to itinerary')
                            open('as.log', 'a').close()
                        except IOError:  # when revvels can't find the init .rst file
                            open('as.log', 'a').write('\nThread ' + thread.basename + ' crashed: initialization did not produce a restart file.')
                            if not restart_on_crash:
                                open('as.log', 'a').write('\nrestart_on_crash = False; thread will not restart')
                                open('as.log', 'a').close()
                            elif restart_on_crash:
                                open('as.log', 'a').write('\nrestart_on_crash = True; resubmitting thread to itinerary')
                                open('as.log', 'a').close()
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
                    if not thread.commit1 and not eps_settings:     # 'and not eps_settings' ensures EPS jobs only get checked for 'commitment' when they exit naturally.
                        thread.commit1 = checkcommit(thread, 'fwd')
                    if not thread.commit2 and not eps_settings:
                        thread.commit2 = checkcommit(thread, 'bwd')
                    if thread.commit1 and thread.jobid1 and not eps_settings:
                        process = subprocess.Popen([cancel_command, thread.jobid1], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()  # doesn't do anything, I think
                        thread.jobid1 = ''
                    if thread.commit2 and thread.jobid2 and not eps_settings:
                        process = subprocess.Popen([cancel_command, thread.jobid2], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()  # doesn't do anything, I think
                        thread.jobid2 = ''
                    if thread.commit1 and thread.commit2:
                        del running[index]
                        index -= 1  # to keep index on track after deleting an entry
                        thread.failcount += 1  # increment fails in a row regardless of outcome
                        if eps_settings and thread.commit1 == 'True' or thread.commit2 == 'True' or (checkcommit(thread, 'init') is True):
                            if not (thread.commit1 == 'fail' or thread.commit2 == 'fail'):
                                thread.last_valid = thread.suffix
                                thread.accept_moves += 1
                                thread.failcount = 0    # reset fail count to zero if this move was accepted
                        elif thread.commit1 != thread.commit2 and thread.commit1 != 'fail' and thread.commit2 != 'fail':  # valid transition path, update 'last_valid' attribute
                            thread.last_valid = thread.suffix
                            thread.accept_moves += 1
                            thread.failcount = 0        # reset fail count to zero if this move was accepted
                        cleanthread(thread)
                index += 1

            pickle.dump(allthreads, open('restart.pkl', 'wb'))  # dump information required to restart ATESA
            if not itinerary:
                time.sleep(60)                                  # delay 60 seconds before checking for job status again

        if not itinerary and not running and not eps_settings:
            acceptances = [(100 * thread.accept_moves / thread.total_moves) for thread in allthreads]
            open('as.log', 'a').write('\nItinerary and running lists are empty.\nAimless shooting is complete! '
                                      'The highest acceptance ratio for any thread was ' + max(acceptances) + '%.'
                                      '\nSee as.out in the working directory for results.')
            open('as.log', 'a').close()
            # todo: add call to LMAX here
            break
        elif not itinerary and not running and eps_settings:
            open('as.log', 'a').write('\nItinerary and running lists are empty.\nEquilibrium path sampling is complete!'
                                      '\nSee eps_results.out in the working directory for results.')
            open('as.log', 'a').close()
            # todo: consider adding call to pymbar here to automate that process?
            break


def update_progress(progress, message='Progress'):
    """
    Print a dynamic progress bar to stdout.

    Credit to Brian Khuu from stackoverflow, https://stackoverflow.com/questions/3160699/python-progress-bar

    Parameters
    ----------
    progress : float
        A number between 0 and 1 indicating the fractional completeness of the bar. A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%.
    message : str
        The string to precede the progress bar (so as to indicate what is progressing)

    Returns
    -------
    None

    """
    barLength = 10  # Modify this to change the length of the progress bar
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
    block = int(round(barLength * progress))
    text = "\r" + message + ": [{0}] {1}% {2}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


# Define runtime for when this program is called
if __name__ == '__main__':
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
    input_file_lines = [i.strip('\n').split(' ') for i in input_file.readlines() if i]  # if i skips blank lines
    input_file.close()

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
    literal_ops = False                             # Determines whether to interpret candidateops as literal python code separated by semi-colons. This isn't a user-defined option, it's handled automatically based on the format of candidateops
    commit_define_fwd = []                          # Defines commitment to fwd basin
    commit_define_bwd = []                          # Defines commitment to bwd basin
    committor_analysis = ''                         # Variables to pass into committor analysis.
    restart = False                                 # Whether or not to restart an old AS run located in working_directory
    groupfile = 0                                   # Number of jobs to submit in one groupfile, if necessary
    groupfile_max_delay = 3600                      # Time in seconds to allow groupfiles to remain in construction before submitting
    include_qdot = False                            # Flag to include order parameter rate of change values in output
    eps_settings = []                               # Equilibrium Path Sampling: [n_windows, k_beads, rc_min, rc_max, traj_length]
    eps_dynamic_seed = ''                           # Flag to seed empty windows using beads from other trajectories during EPS
    minmax_error_behavior = 'exit'                  # Indicates behavior when an reduced OP value is outside range [0,1]
    # todo: add option for solver other than sander.MPI (edit templates to accomodate arbitrary solver name including .MPI if desired)
    # todo: (ambitious?) add support for arbitrary MD engine? I'm unsure whether I think this task is trivial or monumental

    if type(working_directory) == list:             # handles inconsistency in format when the default value is used vs. when a value is given
        working_directory = working_directory[0]


    def str2bool(var):                              # Function to convert string "True" or "False" to corresponding boolean
        return str(var).lower() in ['true']         # returns False for anything other than "True" (case-insensitive)


    # Read in variables from the input file; there's probably a cleaner way to do this, but no matter.
    for entry in input_file_lines:
        try:
            null = entry[2]
        except IndexError:
            sys.exit('Error: option ' + entry[0] + ' was supplied without a corresponding value')

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
            if not '/' in entry[2]:                         # interpreting as a subfolder of cwd
                working_directory = os.getcwd() + '/' + entry[2]
            else:                                           # interpreting as an absolute path
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
            try:
                null = isinstance(ast.literal_eval(entry[2])[0],list)
            except SyntaxError:
                literal_ops = True     # not a nested list, implies that this is an explicit definition
                full_entry = ''                         # string to write all the input into
                for index in range(2,len(entry)):       # full input is contained in entry[index] for all index >= 2
                    full_entry += entry[index] + ' '    # add entry[index] element-wise, followed by a space
                full_entry = full_entry[:-1]            # remove trailing ' '
                candidateops = full_entry.split(';')    # each entry should be a string interpretable as an OP
            if not literal_ops:   # only do this stuff if the OPs are not given literally
                if ' ' in entry[2]:
                    sys.exit('Error: candidate_op cannot contain whitespace (\' \') characters')
                candidateops = ast.literal_eval(entry[2])
                if len(candidateops) < 2:
                    sys.exit('Error: candidate_op must have length >= 2')
                if not isinstance(candidateops[0], list) and not len(candidateops) == 2:
                    sys.exit('Error: if defining candidate_op implicitly, exactly two inputs in a list are required')
                if not isinstance(candidateops[0], list) and not isinstance(candidateops[0], str):
                    sys.exit('Error: if defining candidate_op implicitly, the first entry must be a string (including quotes)')
                if not isinstance(candidateops[0], list) and not isinstance(candidateops[1], str):
                    sys.exit('Error: if defining candidate_op implicitly, the second entry must be a string (including quotes)')
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
        elif entry[0] == 'home_directory':
            home_folder = entry[2]
        elif entry[0] == 'always_new':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: always_new must be either True or False')
            always_new = str2bool(entry[2])
        elif entry[0] == 'rc_definition':
            rc_definition = ''
            for i in range(len(entry)-2):
                rc_definition += entry[i+2] + ' '
            rc_definition = rc_definition[:-1]      # remove trailing space
        elif entry[0] == 'rc_minmax':
            if len(entry) > 3:
                sys.exit('Error: rc_minmax cannot contain whitespace (\' \') characters')
            rc_minmax = ast.literal_eval(entry[2])
            if rc_minmax:   # to allow rc_minmax = '', only perform these checks if it's not
                if not len(rc_minmax) == 2:
                    sys.exit('Error: rc_minmax must have two rows')
                for i in range(len(rc_minmax)):
                    if rc_minmax[0][i] and rc_minmax[1][i] and rc_minmax[0][i] >= rc_minmax[1][i]:
                        sys.exit('Error: values in the second row of rc_minmax must be larger than the corresponding values in the first row')
        elif entry[0] == 'committor_analysis':
            if len(entry) > 3:
                sys.exit('Error: committor_analysis cannot contain whitespace (\' \') characters')
            committor_analysis = ast.literal_eval(entry[2])
            if committor_analysis:  # to allow committor_analysis = [], only perform these checks if it's not
                if not len(committor_analysis) == 5 and not len(committor_analysis) == 6:
                    sys.exit('Error: committor_analysis must be of length five or six (yours is of length ' + str(len(committor_analysis)) + ')')
                elif len(committor_analysis) == 5:
                    committor_analysis.append('')   # if it's not supplied, use an empty string for committor_suffix
                for i in range(len(committor_analysis)):
                    if i in [1,2] and type(committor_analysis[i]) not in [float,int]:
                        sys.exit('Error: committor_analysis[' + str(i) + '] must have type float or int, but has type: ' + str(type(committor_analysis[i])))
                    elif i in [0,3,4] and type(committor_analysis[i]) not in [int]:
                        sys.exit('Error: committor_analysis[' + str(i) + '] must have type int, but has type: ' + str(type(committor_analysis[i])))
                    elif i == 5 and type(committor_analysis[i]) not in [str]:
                        sys.exit('Error: committor_analysis[' + str(i) + '] must have type string, but has type: ' + str(type(committor_analysis[i])))
        elif entry[0] == 'eps_settings':
            if len(entry) > 3:
                sys.exit('Error: eps_settings cannot contain whitespace (\' \') characters')
            eps_settings = ast.literal_eval(entry[2])
            if eps_settings:  # to allow eps_settings = [], only perform these checks if it's not
                if not len(eps_settings) == 6:
                    sys.exit('Error: eps_settings must be of length six (yours is of length ' + str(len(eps_settings)) + ')')
                for i in range(len(eps_settings)):
                    if i in [2,3,5] and type(eps_settings[i]) not in [float,int]:
                        sys.exit('Error: eps_settings[' + str(i) + '] must have type float or int, but has type: ' + str(type(eps_settings[i])))
                    elif i in [0,1,4] and type(eps_settings[i]) not in [int]:
                        sys.exit('Error: eps_settings[' + str(i) + '] must have type int, but has type: ' + str(type(eps_settings[i])))
        elif entry[0] == 'restart':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: restart must be either True or False')
            restart = str2bool(entry[2])
        elif entry[0] == 'groupfile':
            try:
                if int(entry[2]) < 0:
                    sys.exit('Error: groupfile must be greater than or equal to 0')
            except ValueError:
                sys.exit('Error: groupfile must be an integer')
            if isinstance(ast.literal_eval(entry[2]), float):
                print('Warning: groupfile was given as a float (' + entry[2] + '), but should be an integer. '
                      'Rounding down to ' + str(int(entry[2])) + ' and proceeding.')
            groupfile = int(entry[2])
            groupfile_list = []  # initialize list of groupfiles
        elif entry[0] == 'groupfile_max_delay':
            try:
                if int(entry[2]) < 0:
                    sys.exit('Error: groupfile_max_delay must be greater than or equal to 0')
            except ValueError:
                sys.exit('Error: groupfile_max_delay must be an integer')
            if isinstance(ast.literal_eval(entry[2]),float):
                print('Warning: groupfile_max_delay was given as a float (' + entry[2] + '), but should be an integer. '
                      'Rounding down to ' + str(int(entry[2])) + ' and proceeding.')
            groupfile_max_delay = int(entry[2])
        elif entry[0] == 'include_qdot':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: include_qdot must be either True or False')
            include_qdot = str2bool(entry[2])
        elif entry[0] == 'eps_dynamic_seed':
            eps_dynamic_seed = ast.literal_eval(entry[2])
            if type(eps_dynamic_seed) not in [list, int]:
                sys.exit('Error: eps_dynamic_seed must be either an integer or a list (yours is of type ' + str(type(entry[2])) + ')')
            empty_windows = []  # initialize empty windows list to support dynamic seeding
        elif entry[0] == 'minmax_error_behavior':
            if not entry[2] in ['exit','skip','accept']:
                sys.exit('Error: minmax_error_behavior must be either exit, skip, or accept (you gave ' + entry[2] + ')')
            minmax_error_behavior = entry[2]

    # Remove trailing '/' from working_directory for compatibility with my code
    if working_directory[-1] == '/':
        working_directory = working_directory[:-1]

    # Initialize jinja2 environment for filling out templates
    if os.path.exists(home_folder + '/' + 'templates'):
        env = Environment(
            loader=FileSystemLoader(home_folder + '/' + 'templates'),
        )
    else:
        sys.exit('Error: could not locate templates folder: ' + home_folder + '/' + 'templates\nSee documentation for '
                 'the \'home_folder\' option.')

    # Call rc_eval if we're doing an rc_definition or committor_analysis run.
    if rc_definition and not (committor_analysis or eps_settings):
        rc_eval.return_rcs(**locals())
        sys.exit('\nCompleted reaction coordinate evaluation run. See ' + working_directory + '/rc_eval.out for results.')
    elif rc_definition and committor_analysis:
        rc_eval.committor_analysis(**locals())
        sys.exit('\nCompleted committor analysis run. See ' + working_directory + '/committor_analysis' + committor_analysis[5] + '/committor_analysis.out for results.')
    elif not rc_definition and committor_analysis:
        sys.exit('Error: committor analysis run requires rc_definition to be defined')

    # Make a working directory
    dirName = working_directory
    if not resample and not restart and overwrite:  # if resample or restart == True, we want to keep our old working directory
        if os.path.exists(dirName):
            shutil.rmtree(dirName)  # delete old working directory
        os.makedirs(dirName)  # make a new one
    elif resample or restart:
        if resample:
            whichone = 'resample'
        else:
            whichone = 'restart'
        if not os.path.exists(dirName):
            sys.exit('Error: ' + whichone + ' = True, but I can\'t find the working directory: ' + dirName)
    elif not overwrite:  # if resample and restart == False and overwrite == False, make sure dirName doesn't exist...
        if os.path.exists(dirName):
            sys.exit('Error: overwrite = False, but working directory ' + dirName + ' already exists. Move it, choose a'
                     ' different working directory, or add option -O to overwrite it.')
        else:
            os.makedirs(dirName)

    if not resample and not restart:
        if if_glob:
            start_name = glob.glob(initial_structure)   # list of names of coordinate files to begin shooting from
        else:
            start_name = [initial_structure]

        for filename in start_name:
            if ' ' in filename:
                sys.exit('Error: one or more input coordinate filenames contains a space character, which is not '
                         'supported\nThis first offending filename found was: ' + filename)

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

    os.chdir(working_directory)  # move to working directory

    # Initialize files needed for an EPS run if that's what we're doing.
    # Have to do this before potentially calling main_loop() for a restart run if we want that to work.
    # eps_settings = [n_windows, k_beads, rc_min, rc_max, traj_length]
    if eps_settings:
        n_windows = eps_settings[0]
        k_beads = eps_settings[1]
        rc_min = eps_settings[2]
        rc_max = eps_settings[3]
        traj_length = eps_settings[4]
        overlap = eps_settings[5]   # used when setting eps_min/max values, not in defining eps_windows
        if traj_length % k_beads:
            sys.exit('Error: in eps_settings, traj_length is not divisible by k_beads')
        eps_in_template = 'eps_in.tpl'
        for k in range(k_beads-1):
            template = env.get_template(eps_in_template)
            filled = template.render(nstlim=str(int((k + 1) * traj_length / k_beads)),
                                     ntwx=str(int(traj_length / k_beads)))
            with open(working_directory + '/eps' + str(k + 1) + '.in', 'w') as newfile:
                newfile.write(filled)
                newfile.close()
        window_width = (rc_max - rc_min)/n_windows
        eps_windows = [rc_min + window_width*n for n in range(n_windows)] + [rc_min + window_width*(n_windows)]
        if not restart:     # since if restart, this file should already exist
            open('eps_results.out', 'w').write('Lower boundary of RC window; Upper boundary; RC value\n')
            open('eps_results.out', 'a').close()
        elif not os.path.exists('eps_results.out'):
            sys.exit('Error: restart = True and this is an EPS run, but eps_results.out was not found in the working '
                     'directory: ' + working_directory)

    # Check for incompatible options and return helpful error message if a conflict is found
    if restart and (resample or committor_analysis or (rc_definition and not eps_settings)):
        problem = ''
        if resample:
            problem = 'resample = True'
        elif (rc_definition and not eps_settings):
            problem = 'rc_definition is defined and eps_settings is not'
        elif committor_analysis:
            problem = 'committor_analysis is defined'
        elif eps_settings:
            problem = 'restarting an EPS run is not yet supported'  # todo: support it.
        sys.exit('Error: the following options are incompatible: restart = True and ' + problem)
    elif restart:
        # First, carefully load in the necessary information:
        # try:
        #     os.chdir(working_directory)
        # except IOError:
        #     sys.exit('Error: restart = True, but I cannot find the working directory: ' + working_directory)
        try:
            allthreads = pickle.load(open('restart.pkl', 'rb'))
            restartthreads = allthreads
        except IOError:
            sys.exit('Error: restart = True, but I cannot read restart.pkl inside working directory: ' + working_directory)
        # If we're restarting an EPS run, we need to ensure that the EPS settings are compatible. Specifically, we need
        # to ensure that the window boundaries that were previously used still exist in the new ones, and that the
        # number of beads per thread is the same. We would also like to check that the traj_length is the same ideally,
        # but this information is not stored in threads.
        if eps_settings:
            for thread in allthreads:
                if ('%.3f' % (thread.rc_min + overlap) not in ['%.3f' % x for x in eps_windows]) or ('%.3f' % (thread.rc_max - overlap) not in ['%.3f' % x for x in eps_windows]):
                    if ('%.3f'%(thread.rc_min + overlap) not in ['%.3f'%(x) for x in eps_windows]):
                        offending = str(thread.rc_min + overlap)
                    else:
                        offending = str(thread.rc_max - overlap)
                    sys.exit('Error: attempted to restart this EPS run, but the restart.pkl file contained threads with'
                             ' different reaction coordinate boundaries than those defined by the eps_settings option '
                             'for this job. The offending cutoff value (not including overlap) was: ' + offending)
                elif not thread.eps_fwd + thread.eps_bwd + 1 == k_beads:
                    sys.exit('Error: attempted to restart this EPS run, but the restart.pkl file contained threads with'
                             ' a different number of beads than specified by the eps_settings option for this job. '
                             'The restart.pkl threads contain ' + str(thread.eps_fwd + thread.eps_bwd + 1) + ' beads, '
                             'whereas for this run k_beads was set to: ' + str(k_beads))
            if eps_dynamic_seed:
                # Need to handle empty_windows, since unlike in normal behavior threads are beginning (restarting)
                # without going through spawnthread().
                if type(eps_dynamic_seed) == int:
                    eps_dynamic_seed = [eps_dynamic_seed for null in range(len(eps_windows) - 1)]
                elif not len(eps_dynamic_seed) == (len(eps_windows) - 1):
                    sys.exit('Error: eps_dynamic_seed was given as a list, but is not of the same length as the '
                             'number of EPS windows. There are ' + str((len(eps_windows) - 1)) + ' EPS windows but'
                             ' eps_dynamic_seed is of length ' + str(len(eps_dynamic_seed)))
                window_index = 0
                for window in range(len(eps_windows) - 1):
                    empty_windows.append(eps_dynamic_seed[window_index])  # meaning eps_dynamic_seed[window_index] more threads need to start here before it will no longer be considered "empty"
                    window_index += 1
                for thread in allthreads:
                    if thread.status not in ['max_accept', 'max_moves', 'max_fails']:
                        window_index = ['%.3f' % x for x in eps_windows].index('%.3f' % (thread.rc_min + overlap))
                        empty_windows[window_index] -= 1
                        if empty_windows[window_index] < 0:
                            empty_windows[window_index] = 0
                            restartthreads.remove(thread)   # so as not to restart more threads than requested
        running = []
        itinerary = []
        # Next, add those threads that haven't terminated to the itinerary and call main_loop
        for thread in restartthreads:
            if thread.status not in ['max_accept', 'max_moves', 'max_fails']:
                itinerary.append(thread)
        localtime = time.localtime()
        mins = str(localtime.tm_min)
        if len(mins) == 1:
            mins = '0' + mins
        secs = str(localtime.tm_sec)
        if len(mins) == 1:
            secs = '0' + secs
        open('as.log', 'a').write('\n~~~Restarting ' + str(localtime.tm_year) + '-' + str(localtime.tm_mon) + '-' +
                                  str(localtime.tm_mday) + ' ' + str(localtime.tm_hour) + ':' + mins + ':' + secs +
                                  '~~~')
        main_loop()
        sys.exit()

    if eps_settings and (resample or always_new):
        if resample:
            sys.exit('Error: the following options are incompatible: eps_settings and resample')
        if always_new:
            sys.exit('Error: the following options are incompatible: eps_settings and always_new')

    # todo: ### Call to LMAX here (assuming LMAX input file option is supplied) ###
    # Nothing required for this, just put a call to it inside an "if" statement checking for an option in the input file
    # that says "I'm doing LMAX". I'll handle that.
    # LMAX can make its own output file(s), and it should write them to a directory passed in as a string here.

    # Check if candidate_op is given explicitly or as [mask, coordinates], and if the latter, build the explicit form
    # NOTE: This code works and remains in place here for posterity; however, actually using it is probably not advised,
    # as it usually produces far too many OPs to practically test with LMAX.
    if not isinstance(candidateops[0],list) and not literal_ops:
        try:
            traj = pytraj.iterload(candidateops[1], topology)
        except RuntimeError:
            sys.exit('Error: unable to load coordinate file ' + candidateops[1] + ' with topology ' + topology)
        traj.top.set_reference(traj[0])         # set reference frame to first frame
        atom_indices = list(pytraj.select(candidateops[0],traj.top))
        if not atom_indices:                    # if distance = 0 or if there's a formatting error
            sys.exit('Error: found no atoms matching ' + candidateops[0] + ' in file ' + candidateops[1]
                     + '. This may indicate an issue with the format of the candidate_op option!')
        elif len(atom_indices) == 1:            # if formatted correctly but only matches self
            sys.exit('Error: mask ' + candidateops[0] + ' with file ' + candidateops[1] + ' only matches one atom.'
                     + ' Cannot produce any order parameters!')

        # Super-ugly nested loops to build every combination of indices...
        temp_ops = [[],[],[],[]]                # temporary list to append candidate ops to as they're built
        count = 0
        for index in atom_indices:
            count += 1
            update_progress(count/len(atom_indices), 'Building all possible order parameters using atoms matching ' +
                            candidateops[1] + ' with file ' + candidateops[0])
            for second_index in atom_indices:   # second-order connections (distances)
                if not second_index == index:
                    temp_ops[0].append('@' + str(index))
                    temp_ops[1].append('@' + str(second_index))
                    temp_ops[2].append('')
                    temp_ops[3].append('')
                    for third_index in atom_indices:
                        if not index == third_index and not second_index == third_index:
                            temp_ops[0].append('@' + str(index))
                            temp_ops[1].append('@' + str(second_index))
                            temp_ops[2].append('@' + str(third_index))
                            temp_ops[3].append('')
                            # Temporarily commented out to omit dihedrals while we develop LMAX code to handle huge numbers of OPs
                            # for fourth_index in atom_indices:
                            #     if not index == fourth_index and not second_index == fourth_index and not third_index == fourth_index:
                            #         temp_ops[0].append('@' + str(index))
                            #         temp_ops[1].append('@' + str(second_index))
                            #         temp_ops[2].append('@' + str(third_index))
                            #         temp_ops[3].append('@' + str(fourth_index))

        # Then go back through and remove redundant OPs...
        # For some reason this takes much longer than the above block. Reading temp_ops must be much slower than writing?
        position = 0
        count = -1
        total_iters = len(temp_ops[0])
        mirrors_list = []                       # list containing mirrors of OP definitions
        # If an OP is already in mirrors_list, it's deleted from temp_ops; if not, its mirror is added to temp_ops
        while position < len(temp_ops[0]):
            count += 1
            update_progress(count/total_iters, 'Removing redundant coordinates')
            one = temp_ops[0][position]
            two = temp_ops[1][position]
            three = temp_ops[2][position]
            four = temp_ops[3][position]
            if (four and [one,two,three,four] in mirrors_list)\
                    or (three and not four and [one,two,three] in mirrors_list)\
                    or (not three and not four and [one,two] in mirrors_list):
                del temp_ops[0][position]
                del temp_ops[1][position]
                del temp_ops[2][position]
                del temp_ops[3][position]
                position -= 1
            elif four:
                mirrors_list.append([four,three,two,one])
            elif three:
                mirrors_list.append([three,two,one])
            else:
                mirrors_list.append([two,one])
            position += 1

        candidateops = temp_ops
        print('\nThe finalized order parameter definition is: ' + str(candidateops))

    # Return an error and exit if the input file is missing entries for non-optional inputs.
    if 'commit_fwd' not in [entry[0] for entry in input_file_lines] and not (resample or rc_definition or eps_settings):
        sys.exit('Error: input file is missing entry for commit_fwd, which is non-optional')
    if 'commit_bwd' not in [entry[0] for entry in input_file_lines] and not (resample or rc_definition or eps_settings):
        sys.exit('Error: input file is missing entry for commit_bwd, which is non-optional')
    if 'candidate_op' not in [entry[0] for entry in input_file_lines]:
        sys.exit('Error: input file is missing entry for candidate_op, which is non-optional')

    if not resample:
        itinerary = []                  # a list of threads that need running
        running = []                    # a list of currently running threads
        allthreads = []                 # a list of all threads regardless of status

        try:
            os.remove('as.log')                 # delete previous run's log
        except OSError:                         # catches error if no previous log file exists
            pass
        with open('as.log', 'w+') as newlog:
            localtime = time.localtime()
            mins = str(localtime.tm_min)
            if len(mins) == 1:
                mins = '0' + mins
            secs = str(localtime.tm_sec)
            if len(mins) == 1:
                secs = '0' + secs
            newlog.write('~~~New log file ' + str(localtime.tm_year) + '-' + str(localtime.tm_mon) + '-' +
                          str(localtime.tm_mday) + ' ' + str(localtime.tm_hour) + ':' + mins + ':' + secs + '~~~')
            newlog.close()

        for structure in start_name:                # for all of the initial structures...
            shutil.copy(called_path + '/' + structure, './')        # copy the input structure to the working directory...
            try:
                shutil.copy(called_path + '/' + topology, './')     # ... and its little topology file, too!
            except OSError:
                sys.exit('Error: could not find the indicated topology file: ' + called_path + '/' + topology)
            thread = spawnthread(structure,suffix='1')              # spawn a new thread with the default settings
            allthreads.append(thread)                               # add it to the list of all threads for bookkeeping
            thread.last_valid = '0'                                 # so that if the first shooting point does not result in a valid transition path, shooting will begin from the TS guess
            thread.prmtop = topology                                # set prmtop filename for the thread
            itinerary.append(thread)                                # submit it to the itinerary
            if degeneracy > 1:                                      # if degeneracy > 1, files in start_name were copies...
                os.remove(called_path + '/' + structure)            # ... delete them to keep the user's space clean!

    try:
        os.remove('as.out')                 # delete previous run's output file
    except OSError:                         # catches error if no previous output file exists
        pass
    with open('as.out', 'w+') as newout:    # make a new output file
        newout.close()

    # Implementation of resample
    if resample:                            # if True, this is a resample run, so we'll head off the simulations steps here
        pattern = re.compile('\ .*\ finished')                              # pattern to find job name
        pattern2 = re.compile('result:\ [a-z]*\ ')                          # pattern for basin commitment flag
        try:
            logfile = open('as.log')                                        # open log for reading...
        except OSError:
            sys.exit('Error: could not find as.log in working directory: ' + working_directory)
        logfile_lines = logfile.readlines()
        logfile.close()
        count = 0
        for line in logfile_lines:                                          # iterate through log file
            if 'finished with fwd trajectory result: ' in line:             # looking for lines with results # todo: decide whether there's good reason not to include bwd trajectory results as well
                commit = pattern2.findall(line)[0][8:-1]                    # first, identify the commitment flag
                if commit != 'fail':                                        # none of this matters if it was "fail"
                    basin = 'error'
                    if commit == 'fwd':
                        basin = 'A'
                    elif commit == 'bwd':
                        basin = 'B'
                    init_name = pattern.findall(line)[0][5:-9] + '_init_fwd.rst' # clunky; removes "run " and " finished"
                    open('as.out', 'a').write(basin + ' <- ' + candidatevalues(init_name) + '\n')
                    open('as.log', 'a').close()
            count += 1
            update_progress(count / len(logfile_lines), 'Resampling by searching through logfile')
        # todo: add call to LMAX here as well
        sys.exit('Resampling complete; written new as.out')

    os.makedirs('history')                  # make a new directory to contain the history files of each thread

    main_loop()
