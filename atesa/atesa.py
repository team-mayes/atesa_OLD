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
import numpy
from jinja2 import Environment, FileSystemLoader
#from atesa import rc_eval                                              # this one I think might not work at all
try:
    rc_eval = importlib.import_module('.rc_eval', package=__package__)  # this one works with tests
except TypeError:
    rc_eval = importlib.import_module('rc_eval')                        # this one works to run code, but breaks tests


# Define the "thread" object that constitutes one string of shooting moves in our search through phase space. Each
# thread is an independent series of simulations with its own acceptance ratio, and may or may not have a unique
# starting structure.
class Thread(object):           # add (object) explicitly to force compatibility of restart pickle file with Python 2
    def __init__(self, settings=argparse.Namespace()):
        self.basename = ''      # first part of name, universal to all files in the thread
        self.name = ''          # full name, identical to basename unless otherwise specified
        self.suffix = ''        # appended to basename after an underscore to build name. Should be str(an integer)
        self.jobid1 = ''        # current batch system jobid associated with the thread init step or forward trajectory
        self.jobid2 = ''        # another jobid slot, for the backward trajectory
        self.type = ''          # thread job type; init to get initial trajectories or prod to run forward and backward
        self.start_name = ''    # name of previous thread to initialize shooting from
        self.last_valid = '0'   # suffix of the most recent shooting point that resulted in a valid transition path
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
            if settings.eps_settings:
                self.eps_fwd = random.randint(0,settings.eps_settings[1] - 1)    # number of beads to propagate forward in the next step
                self.eps_bwd = settings.eps_settings[1] - self.eps_fwd - 1       # number of beads to propagate backward in the next step
                self.eps_fwd_la = self.eps_fwd                          # value from the last accepted move
                self.eps_bwd_la = self.eps_bwd                          # value from the last accepted move
                self.rc_min = 0     # lower boundary of RC window that this thread started in
                self.rc_max = 0     # upper boundary of RC window that this thread started in
        except:
            pass    # allows Thread() to be called without requiring eps_settings to have been defined yet


# Then, define a series of functions to perform the basic tasks of creating batch files, interacting with the batch
# system, and interpreting/modifying coordinate and trajectory files.
def handle_groupfile(job_to_add='',settings=argparse.Namespace):
    """
    Update attributes of settings.groupfile_list and return name of the groupfile that should be written to next

    Routine to handle groupfiles when necessary (i.e., when groupfile > 0). Maintains a list object keeping track of
    existing groupfiles and their contents as well as their statuses and ages. This function is called in makebatch()
    to return the name of the groupfile that that function should be writing to, and subbatch() redirects here when
    groupfile > 0. Finally, it is called at the end of every main_loop loop to update statuses as needed.

    job_to_add is an optional argument; if present, this function appends the name of the job given to the <CONTENTS>
    field for the groupfile that it returns. This puts the onus on the code that calls handle_groupfile() to actually
    put that job in that groupfile, of course!

    Process flow for this function:
      1. iterate through existing list of groupfiles as read from settings.groupfile_list
      2. update statuses as necessary
      3. submit any of them that have status "construction" and length == settings.groupfile, or
         age > settings.groupfile_max_delay and length >= 1 (where length means number of lines)
      4. if the list no longer contains any with status "construction", make a new, blank one and return its name
      5. otherwise, return the name of the groupfile with status "construction"

    Format of each sublist of settings.groupfile_list:
      [<NAME>, <STATUS>, <CONTENTS>, <TIME>]
          <NAME> is the name of the groupfile
          <STATUS> is any of:
              "construction", meaning this groupfile is the one to add new lines to (only one of these at a time),
              "completed", meaning the groupfile was submitted and has since terminated,
              "processed", meaning the groupfile has been fed through main_loop and followup jobs submitted, or
              a string of numbers indicating a jobid of a currently-running groupfile
          <CONTENTS> is a list of the jobs contained in the groupfile (their (thread.name + '_' + thread.type) values)
          <TIME> is the time in seconds when the groupfile was created
              if the current time - TIME > settings.groupfile_max_delay, then the groupfile is submitted regardless of length
              (assuming it's at least of length 1)

    Parameters
    ----------
    job_to_add : str
        Name of job to append to the current groupfile's settings.groupfile_list entry. Default = '' (don't add any jobs)
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    str
        Name of the groupfile to which new jobs should be appended.

    """

    def new_groupfile(job_to_add_local):
        # Specialized subroutine to build a new empty groupfile and add its information to the settings.groupfile_list
        # If a job_to_add is present, add it to the new settings.groupfile_list entry.
        if settings.groupfile_list:
            pattern = re.compile('\_[0-9]*')
            suffix = pattern.findall(settings.groupfile_list[-1][0])[0][1:]  # this gets the suffix of the last groupfile
            suffix = str(int(suffix) + 1)
        else:
            suffix = '1'
        open('groupfile_' + suffix,'w').close()                     # make a new, empty groupfile
        settings.groupfile_list.append(['groupfile_' + suffix,'construction',job_to_add_local,time.time()])
        return 'groupfile_' + suffix

    def sub_groupfile(groupfile_name):
        # Specialized subroutine to submit a groupfile job as a batch job
        # Creates a batchfile from the template and submits, returning jobid
        batch = 'batch_' + settings.batch_system + '_groupfile.tpl'
        open(settings.logfile, 'a').write('\nWriting groupfile batch file for ' + groupfile_name)
        open(settings.logfile, 'a').close()

        # This block necessary to submit jobs with one fewer than max groups when "flag" from outer scope is True
        if flag:
            groupcount = settings.groupfile - 1
            this_ppn = int(settings.prod_ppn - (settings.prod_ppn/settings.groupfile))  # remove cores proportional to removed simulations
        else:
            groupcount = settings.groupfile
            this_ppn = settings.prod_ppn

        template = settings.env.get_template(batch)
        filled = template.render(nodes=settings.prod_nodes, taskspernode=settings.this_ppn, walltime=settings.prod_walltime, solver='sander',
                                 mem=settings.prod_mem, working_directory=settings.working_directory, groupcount=groupcount,
                                 groupfile=groupfile_name, name=groupfile_name)
        with open(groupfile_name + '_groupfile.' + settings.batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

        output = interact(groupfile_name + '_groupfile.' + settings.batch_system, settings=settings)
        return output


    for groupfile_data in settings.groupfile_list:
        try:                                            # if groupfile was running when this function was last called
            null = int(groupfile_data[1])               # cast to int only works if status is a jobid
            # Check queue to see if it's still running
            output = interact('queue',settings)

            if not groupfile_data[1] in str(output):         # all jobs in the groupfile have finished
                groupfile_data[1] = 'completed'         # update job status
        except ValueError:
            if groupfile_data[1] == 'construction':     # groupfile is the one being built
                # First, check if this groupfile is due to be submitted based on settings.groupfile_max_delay
                age = time.time() - groupfile_data[3]   # age of groupfile entry
                length = len(open(groupfile_data[0],'r').readlines())
                # flag helps us know to submit jobs a little early when there are only prod jobs in the itinerary and
                # no room for both halves of another prod job in the groupfile todo: won't work with committor analysis
                flag = (length == settings.groupfile - 1 and 'init' not in [thread.type for thread in settings.itinerary])
                if length >= 1 and ((age > settings.groupfile_max_delay > 0) or length == settings.groupfile or flag):
                    groupfile_data[1] = sub_groupfile(groupfile_data[0])    # submit groupfile with call to sub_groupfile()
                    return new_groupfile(job_to_add)    # make a new groupfile to return
                else:
                    if job_to_add:                      # add the job_to_add to the contents field, if applicable
                        groupfile_data[2] += job_to_add + ' '
                    return groupfile_data[0]            # this is the current groupfile and it's not full, so return it

    if not settings.groupfile_list:                          # this is the first call to this function, so no groupfiles exist yet
        return new_groupfile(job_to_add)            # make a new groupfile to return


def interact(type, settings):
    """
    Handle submitting of jobs to the batch system or looking up of currently queued and running jobs

    Parameters
    ----------
    type : str
        Two valid options:
        - "queue", to get currently queued and running jobs, or
        - anything else, to submit the batch file given by type to the batch system
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    str
        output from batch system depending on the input:
        - "queue", returns raw output
        - anything else, returns the jobid number of the newly submitted batch job (as a string)

    """
    if settings.DEBUGMODE:
        if type == 'queue':
            # Just return an empty qstat string
            return 'JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n'
        else:
            sys.exit('Error: interact() does not support DEBUGMODE with (not type == \'queue\')')

    user_alias = '$USER'
    command = ''        # line just here to suppress "variable may be called before being defined" error in my IDE
    if type == 'queue':
        if settings.batch_system == 'pbs':
            command = 'qselect -u ' + user_alias + ' -s QR'
        elif settings.batch_system == 'slurm':
            command = 'squeue -u ' + user_alias
    else:               # submitting the batch file given by "type"
        if settings.batch_system == 'pbs':
            command = 'qsub ' + str(type)
        elif settings.batch_system == 'slurm':
            command = 'sbatch ' + str(type)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               close_fds=True, shell=True)
    output = process.stdout.read()
    # Some PBS-specific error handling to help handle common issues by simply resubmitting as necessary.
    while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read()
    if 'Bad UID for job execution MSG=user does not exist in server password file' in str(output) or\
       'This stream has already been closed. End of File.' in str(output):
        open(settings.logfile, 'a').write('\nWarning: attempted to submit a job, but got error: ' + str(output) + '\n'
                                  + 'Retrying in one minute...')
        open(settings.logfile, 'a').close()
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
            open(settings.logfile, 'a').write('\nBatch system says: ' + str(output))
            open(settings.logfile, 'a').close()
            return pattern.findall(str(output))[0]
        except IndexError:
            sys.exit('Error: unable to submit a batch job: ' + str(type) + '. Got message: ' + str(output))


def makebatch(thread, settings):
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
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    new_file : str
        Name of the newly written file. If more than one file was written, this is only the most recent one.

    """
    name = thread.name
    type = thread.type
    batch = 'batch_' + settings.batch_system + '.tpl'

    if not os.path.exists(settings.home_folder + '/' + 'input_files'):
        sys.exit('Error: could not locate input_files folder: ' + settings.home_folder + '/' + 'input_files\n'
                  'See documentation for the \'home_folder\' option.')

    if settings.eps_settings:
        prod_fwd = settings.working_directory + '/eps' + str(thread.eps_fwd) + '.in'
        prod_bwd = settings.working_directory + '/eps' + str(thread.eps_bwd) + '.in'
    else:
        prod_fwd = settings.home_folder + '/input_files/prod.in'
        prod_bwd = settings.home_folder + '/input_files/prod.in'

    # Append a new line to the current groupfile (this is spiritually similar to building a batch file for jobs that are
    # submitted individually; the actual batch file to submit the group file is built by handle_groupfile() when ready)
    # This implementation is ugly because groupfile support was added later in development. If I ever get the chance to
    # go back through and rewrite this code, this is an opportunity for polish!
    if settings.groupfile > 0:
        current_groupfile = handle_groupfile(thread.name + '_' + thread.type, settings)
        if type == 'init':
            inp = settings.home_folder + '/input_files/init.in'
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
            for i in range(settings.n_shots):
                inp = settings.home_folder + '/input_files/committor_analysis.in'
                out = name + '_ca_' + str(i) + '.out'
                top = '../' + thread.prmtop
                inpcrd = '../' + name
                rst = name + '_ca_' + str(i) + '.rst'
                nc = name + '_ca_' + str(i) + '.nc'
                open(current_groupfile, 'a').write('-i ' + inp + ' -o ' + out + ' -p ' + top + ' -c ' + inpcrd + ' -r ' + rst + ' -x ' + nc + '\n')
                open(current_groupfile, 'a').close()
                # Since we're adding multiple lines here, we need to check at every iteration if a new groupfile is
                # required. handle_groupfile() handles this for us neatly.
                if len(open(current_groupfile, 'r').readlines()) == settings.groupfile:
                    current_groupfile = handle_groupfile(name + '_ca_' + str(i), settings)
        return ''   # do not proceed with the remainder of this function

    if type == 'init':
        # init batch file
        open(settings.logfile, 'a').write('\nWriting init batch file for ' + name + ' starting from ' + thread.start_name)
        open(settings.logfile, 'a').close()
        template = settings.env.get_template(batch)
        filled = template.render(name=name + '_init', nodes=settings.init_nodes, taskspernode=settings.init_ppn, walltime=settings.init_walltime,
                                 solver='sander', inp=settings.home_folder + '/input_files/init.in', out=name + '_init.out',
                                 prmtop=thread.prmtop, inpcrd=thread.start_name, rst=name + '_init_fwd.rst',
                                 nc=name + '_init.nc', mem=settings.init_mem, working_directory=settings.working_directory,
                                 self_name=name + '_init.' + settings.batch_system)
        with open(name + '_init.' + settings.batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

    elif type == 'prod':
        # forward and backward simulation batch files
        open(settings.logfile, 'a').write('\nWriting forward batch file for ' + name)
        open(settings.logfile, 'a').close()
        template = settings.env.get_template(batch)
        filled = template.render(name=name + '_fwd', nodes=settings.prod_nodes, taskspernode=settings.prod_ppn, walltime=settings.prod_walltime,
                                 solver='sander', inp=prod_fwd, out=name + '_fwd.out', prmtop=thread.prmtop,
                                 inpcrd=name + '_init_fwd.rst', rst=name + '_fwd.rst', nc=name + '_fwd.nc',
                                 mem=settings.prod_mem, working_directory=settings.working_directory,
                                 self_name=name + '_fwd.' + settings.batch_system)
        with open(name + '_fwd.' + settings.batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

        open(settings.logfile, 'a').write('\nWriting backward batch file for ' + name)
        open(settings.logfile, 'a').close()
        template = settings.env.get_template(batch)
        filled = template.render(name=name + '_bwd', nodes=settings.prod_nodes, taskspernode=settings.prod_ppn, walltime=settings.prod_walltime,
                                 solver='sander', inp=prod_bwd, out=name + '_bwd.out', prmtop=thread.prmtop,
                                 inpcrd=name + '_init_bwd.rst', rst=name + '_bwd.rst', nc=name + '_bwd.nc',
                                 mem=settings.prod_mem, working_directory=settings.working_directory,
                                 self_name=name + '_bwd.' + settings.batch_system)
        with open(name + '_bwd.' + settings.batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()

    elif type == 'committor_analysis':
        name = thread.basename
        for i in range(settings.n_shots):
            template = settings.env.get_template(batch)
            filled = template.render(name=name + '_ca_' + str(i), nodes=settings.prod_nodes, taskspernode=settings.prod_ppn,
                                     walltime=settings.prod_walltime, solver='sander',
                                     inp=settings.home_folder + '/input_files/committor_analysis.in',
                                     out=name + '_ca_' + str(i) + '.out', prmtop='../' + thread.prmtop,
                                     inpcrd='../' + name, rst=name + '_ca_' + str(i) + '.rst',
                                     nc=name + '_ca_' + str(i) + '.nc', mem=settings.prod_mem,
                                     working_directory=settings.working_directory + '/committor_analysis' + settings.committor_analysis[5],
                                     self_name=name + '_ca_' + str(i) + '.' + settings.batch_system)
            with open(name + '_ca_' + str(i) + '.' + settings.batch_system, 'w') as newfile:
                newfile.write(filled)
                newfile.close()
            open(settings.logfile, 'a').write('\nWriting committor analysis batch file with initial point: ' + name + ' and '
                                      'suffix: ' + str(i))

    elif type == 'bootstrap':
        for i in range(settings.bootstrap_n + 1):
            name = 'bootstrap_' + str(settings.len_data) + '_' + str(i)
            template = settings.env.get_template('lmax_' + settings.batch_system + '.tpl')
            filled = template.render(name=name, nodes=1, taskspernode=1, walltime=settings.prod_walltime,
                                     input=settings.working_directory + '/' + name + '.in', output_file=name + '.out',
                                     mem=settings.prod_mem, working_directory=settings.working_directory, k_best='null',
                                     fixed='null', running=str(settings.cvs_in_rc), bootstrap='0',
                                     home_directory=settings.home_folder, self_name=name + '.' + settings.batch_system)
            # k_best and fixed = 'null' because they are ignored since running is given. Here 'bootstrap' refers to a
            # separate bootstrapping algorithm within LMAX that I wrote for something else, not to the RC bootstrapping
            # that it going on when this code is called, so it should be turned off (set to 0).
            with open(name + '.' + settings.batch_system, 'w') as newfile:
                newfile.write(filled)
                newfile.close()
            open(settings.logfile, 'a').write('\nWriting bootstrap batch file: ' + name)

    # This error is obsolete, as unless the code is broken in some way it should be unreachable (job type is not a user-
    # supplied value)
    else:
        open(settings.logfile, 'a').write('\nAn incorrect job type \"' + type + '\" was passed to makebatch.')
        open(settings.logfile, 'a').close()
        sys.exit('An incorrect job type \"' + type + '\" was passed to makebatch.')

    return newfile.name


def subbatch(thread,direction='',logfile='as.log',settings=argparse.Namespace):
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
    logfile : str
        Name of the logfile to write output to. This is used to redirect the output from the default logfile to ca.log
        when subbatch is called from rc_eval.committor_analysis(). Default = 'as.log'
    settings : Namespace
        Global settings Namespace object.

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

    if settings.groupfile > 0:
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

    if settings.DEBUGMODE:
        if not type == 'committor_analysis' and not direction == 'bootstrap':
            # First, check to ensure that the batch file exists
            if not os.path.exists(name + '_' + type + '.' + settings.batch_system):
                sys.exit('Error: subbatch() was called to submit a non-existant batch file: ' + name + '_' + type + '.'
                    + settings.batch_system)
            # Return a randomly generated but non-redundant jobid string
            random_jobid = str(random.randint(0,99999999))
            while random_jobid in [thread.jobid1 for thread in settings.allthreads] or \
                    random_jobid in [thread.jobid2 for thread in settings.allthreads] or \
                    random_jobid in [[jobid for jobid in thread.jobidlist] for thread in settings.allthreads]:
                random_jobid = str(random.randint(0, 99999999))
            return random_jobid
        elif type == 'committor_analysis':
            # As above, but a list of settings.n_shots randomly generated strings
            output = []
            for i in range(settings.n_shots):
                if not os.path.exists(name + '_ca_' + str(i) + '.' + settings.batch_system):
                    sys.exit('Error: subbatch() was called to submit a non-existant batch file: ' + name + '_ca_' +
                             str(i) + '.' + settings.batch_system)
                random_jobid = str(random.randint(0, 99999999))
                while random_jobid in [thread.jobid1 for thread in settings.allthreads] or \
                            random_jobid in [thread.jobid2 for thread in settings.allthreads] or \
                            random_jobid in [[jobid for jobid in thread.jobidlist] for thread in settings.allthreads] or \
                            random_jobid in output:
                    random_jobid = str(random.randint(0, 99999999))
                output.append(random_jobid)
            return output
        else:
            # As above, but for bootstrap-style jobs
            output = []
            for i in range(settings.bootstrap_n + 1):
                if not os.path.exists('bootstrap_' + str(settings.len_data) + '_' + str(i) + '.' + settings.batch_system):
                    sys.exit('Error: subbatch() was called to submit a non-existant batch file: bootstrap_' +
                             str(settings.len_data) + '_' + str(i) + '.' + settings.batch_system)
                random_jobid = str(random.randint(0, 99999999))
                while random_jobid in [thread.jobid1 for thread in settings.allthreads] or \
                                random_jobid in [thread.jobid2 for thread in settings.allthreads] or \
                                random_jobid in [[jobid for jobid in thread.jobidlist] for thread in
                                                 settings.allthreads] or \
                                random_jobid in output:
                    random_jobid = str(random.randint(0, 99999999))
                output.append(random_jobid)
            return output

    if type == 'committor_analysis':    # committor analysis branch
        jobids = []
        for i in range(settings.n_shots):
            open(settings.logfile, 'a').write('\nSubmitting job: ' + name + '_ca_' + str(i) + '.' + settings.batch_system)
            open(settings.logfile, 'a').close()
            output = interact(name + '_ca_' + str(i) + '.' + settings.batch_system,settings)
            jobids.append(output)
        return jobids
    elif direction == 'bootstrap':      # bootstrapping branch
        jobids = []
        for i in range(settings.bootstrap_n + 1):
            open(settings.logfile, 'a').write('\nSubmitting job: ' + 'bootstrap_' + str(settings.len_data) + '_' + str(i) + '.' + settings.batch_system)
            open(settings.logfile, 'a').close()
            output = interact('bootstrap_' + str(settings.len_data) + '_' + str(i) + '.' + settings.batch_system, settings)
            jobids.append(output)
        return jobids
    else:                               # aimless shooting, find_ts, or EPS branch
        open(settings.logfile, 'a').write('\nSubmitting job: ' + name + '_' + type + '.' + settings.batch_system)
        open(settings.logfile, 'a').close()
        output = interact(name + '_' + type + '.' + settings.batch_system, settings=settings)
        return output


def spawnthread(basename, thread_type='init', suffix='', settings=argparse.Namespace()):
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
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    Thread
        The newly created Thread object.

    """
    new_thread = Thread(settings)
    new_thread.basename = basename
    new_thread.suffix = suffix
    new_thread.name = basename + '_' + suffix
    new_thread.type = thread_type
    new_thread.start_name = basename
    new_thread.prmtop = settings.topology

    if not settings.committor_analysis:     # committor analysis calls this function but does not have an allthreads attribute in its settings
        settings.allthreads.append(new_thread)

    if settings.eps_settings:    # need to evaluate the window this Thread started in and store its boundaries.
        try:            # test if eps settings has been initialized yet
            null = settings.eps_windows
        except AttributeError:
            sys.exit('Error: spawnthread() was called with eps_settings = True, but eps_windows has not been initialized')
        cv_values = [float(cv) for cv in candidatevalues(basename,reduce=True,settings=settings).split(' ') if cv]  # OP values as a list
        rc_value = get_rc_value(cv_values=cv_values, settings=settings)

        if settings.eps_dynamic_seed:
            if not settings.empty_windows:   # if empty_windows is empty, i.e., this is the first thread being spawned
                if type(settings.eps_dynamic_seed) == int:
                    settings.eps_dynamic_seed = [settings.eps_dynamic_seed for null in range(len(settings.eps_windows) - 1)]
                elif not len(settings.eps_dynamic_seed) == (len(settings.eps_windows) - 1):
                    sys.exit('Error: eps_dynamic_seed was given as a list, but is not of the same length as the '
                             'number of EPS windows. There are ' + str((len(settings.eps_windows) - 1)) + ' EPS windows but'
                             ' eps_dynamic_seed is of length ' + str(len(settings.eps_dynamic_seed)))
                window_index = 0
                for window in range(len(settings.eps_windows) - 1):
                    settings.empty_windows.append(settings.eps_dynamic_seed[window_index])  # meaning eps_dynamic_seed[window_index] more threads need to start here before it will no longer be considered "empty"
                    window_index += 1

        for window in range(len(settings.eps_windows)-1):
            try:
                if not settings.dynamic_seed_kludge == '':  # see cleanthread
                    window = settings.dynamic_seed_kludge
                    settings.dynamic_seed_kludge = ''
            except AttributeError:
                pass    # this attribute not having been defined yet is fine
            # use of inclusive inequality on both sides prevents values equal to the min or max from presenting issues
            if settings.eps_windows[window] - settings.overlap <= rc_value <= settings.eps_windows[window+1] + settings.overlap:
                open(settings.logfile, 'a').write('\nCreating new thread starting from initial structure ' + basename + ' with'
                                          ' RC value ' + str(rc_value))
                new_thread.rc_min = settings.eps_windows[window] - settings.overlap
                new_thread.rc_max = settings.eps_windows[window+1] + settings.overlap
                if settings.eps_dynamic_seed:
                    settings.empty_windows[window] -= 1   # set empty_windows for this window to 1 less
                    if settings.empty_windows[window] < 0:
                        settings.empty_windows[window] = 0
                break
        if not new_thread.rc_min and not new_thread.rc_max:     # since if I check just one, it could be zero!
            sys.exit('Error: initial structure ' + basename + ' has RC value (' + str(rc_value) + ') outside the '
                     'defined range (' + str(settings.rc_min) + ' to ' + str(settings.rc_max) + ').')

    return new_thread


def checkcommit(thread,direction,directory='',settings=argparse.Namespace()):
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
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    str
        Either 'fwd' when the commit_fwd criteria are met; 'bwd' when the commit_bwd criteria are met; or '' when
        neither are met. Alternatively, 'eps' if eps_settings was provided (no computations are performed in this case)

    """
    committor_directory = ''            # empty if directory isn't given, so we're looking in working_directory
    if directory:
        directory += '/'
        committor_directory = directory + 'committor_analysis' + settings.committor_analysis[5] + '/'

    if direction not in ['fwd','bwd','init']:  # occurs when this is called by rc_eval.py or init_find_ts
        name = thread   # in this case 'thread' should be a string, not a Thread() object. This is sloppy.
        if not os.path.isfile(committor_directory + name):  # if the file doesn't exist yet, just do nothing
            return ''
        if not settings.eps_settings:
            traj = pytraj.iterload(committor_directory + name, directory + settings.topology, format='.nc')
    else:
        name = thread.name
        if not os.path.isfile(committor_directory + name + '_' + direction + '.nc'):  # if the file doesn't exist yet, just do nothing
            return ''
        if not settings.eps_settings:
            traj = pytraj.iterload(committor_directory + name + '_' + direction + '.nc', directory + settings.topology,
                                   format='.nc')

    commit_flag = ''                    # initialize flag for commitment; this is the value to be returned eventually
    # If we're doing EPS, the commitment criterion is that any of the beads reside inside the given range of RC values
    if settings.eps_settings:
        if direction == 'init':
            filename = thread.name + '_init_fwd.rst'
            traj = pytraj.iterload(filename, directory + settings.topology, format='.rst7')
        elif direction in ['fwd','bwd']:
            filename = name + '_' + direction + '.nc'
            traj = pytraj.iterload(filename, directory + settings.topology, format='.nc')
        else:
            sys.exit('Error: checkcommit() was passed an incorrect direction argument: ' + str(direction))
        rc_values = []
        for i in range(traj.__len__()):                         # iterate through frames
            cv_values = [float(cv) for cv in candidatevalues(filename, frame=i, reduce=True, settings=settings).split(' ') if cv]  # CV values as a list
            rc_values.append(get_rc_value(cv_values=cv_values, settings=settings))
        commit_flag = 'False'           # return 'False' if no beads are in bounds
        for value in rc_values:
            if thread.rc_min <= value <= thread.rc_max:
                commit_flag = 'True'    # return 'True' if any of the beads are in bounds.
                break
    else:   # normal aimless shooting behavior
        if not traj:  # catches error if the trajectory file exists but has zero frames
            print('Don\'t worry about this internal error; it just means that ATESA is checking for commitment in a '
                  'trajectory that doesn\'t have any frames yet, probably because the simulation has only just begun.')
            return ''
        for i in range(0,len(settings.commit_define_fwd[2])):
            if settings.commit_define_fwd[3][i] == 'lt':
                if pytraj.distance(traj, settings.commit_define_fwd[0][i] + ' ' + settings.commit_define_fwd[1][i], n_frames=1)[-1] <= settings.commit_define_fwd[2][i]:
                    commit_flag = 'fwd'     # if a commitor test is passed, testing moves on to the next one.
                else:
                    commit_flag = ''
                    break                   # if a commitor test is not passed, all testing in this direction ends in a fail
            elif settings.commit_define_fwd[3][i] == 'gt':
                if pytraj.distance(traj, settings.commit_define_fwd[0][i] + ' ' + settings.commit_define_fwd[1][i], n_frames=1)[-1] >= settings.commit_define_fwd[2][i]:
                    commit_flag = 'fwd'
                else:
                    commit_flag = ''
                    break
            else:
                open(settings.logfile, 'a').write('\nAn incorrect commitor definition \"' + settings.commit_define_fwd[3][i] + '\" was given for index ' + str(i) +' in the \'fwd\' direction.')
                open(settings.logfile, 'a').close()
                sys.exit('An incorrect commitor definition \"' + settings.commit_define_fwd[3][i] + '\" was given for index ' + str(i) +' in the \'fwd\' direction.')

        if commit_flag == '':                # only bother checking for bwd commitment if not fwd commited
            for i in range(0,len(settings.commit_define_bwd[2])):
                if settings.commit_define_bwd[3][i] == 'lt':
                    if pytraj.distance(traj, settings.commit_define_bwd[0][i] + ' ' + settings.commit_define_bwd[1][i], n_frames=1)[-1] <= settings.commit_define_bwd[2][i]:
                        commit_flag = 'bwd'  # if a commitor test is passed, testing moves on to the next one.
                    else:
                        commit_flag = ''
                        break                # if a commitor test is not passed, all testing in this direction ends in a fail
                elif settings.commit_define_bwd[3][i] == 'gt':
                    if pytraj.distance(traj, settings.commit_define_bwd[0][i] + ' ' + settings.commit_define_bwd[1][i], n_frames=1)[-1] >= settings.commit_define_bwd[2][i]:
                        commit_flag = 'bwd'
                    else:
                        commit_flag = ''
                        break
                else:
                    open(settings.logfile, 'a').write('\nAn incorrect commitor definition \"' + settings.commit_define_bwd[3][i] + '\" was given for index ' + str(i) +' in the \'bwd\' direction.')
                    open(settings.logfile, 'a').close()
                    sys.exit('An incorrect commitor definition \"' + settings.commit_define_bwd[3][i] + '\" was given for index ' + str(i) +' in the \'bwd\' direction.')

    del traj    # to ensure cleanup of iterload object

    return commit_flag


def standalone_checkcommit(name, settings):   # todo: delete or modify this or checkcommit() so only one is needed
    # This is basically just checkcommit() modified to work on individual coordinate files instead of with threads.
    # This is for use with skip_log = True, so unlike in checkcommit, we never want to return a blank commitment flag;
    # this is for analysis only, so if it's not commited to fwd or bwd, then it's failed.

    if not os.path.isfile(settings.working_directory + '/' + name):  # if the file doesn't exist, return 'fail'
        return 'fail'

    traj = pytraj.iterload(settings.working_directory + '/' + name, settings.working_directory + '/' + settings.topology, format='.nc')

    if not traj:                            # catches error if the trajectory file exists but has zero frames
        print('Don\'t worry about this internal error; it\'s from pytraj trying to load an empty trajectory, which ATESA can handle properly.')
        return 'fail'

    commit_flag = ''                        # initialize flag for commitment; this is the value to be returned eventually
    for i in range(0, len(settings.commit_define_fwd[2])):
        if settings.commit_define_fwd[3][i] == 'lt':
            if pytraj.distance(traj, settings.commit_define_fwd[0][i] + ' ' + settings.commit_define_fwd[1][i], n_frames=1)[-1] <= settings.commit_define_fwd[2][i]:
                commit_flag = 'fwd'         # if a committor test is passed, testing moves on to the next one.
            else:
                commit_flag = ''
                break                       # if a committor test is not passed, all testing in this direction ends in a fail
        elif settings.commit_define_fwd[3][i] == 'gt':
            if pytraj.distance(traj, settings.commit_define_fwd[0][i] + ' ' + settings.commit_define_fwd[1][i], n_frames=1)[-1] >= settings.commit_define_fwd[2][i]:
                commit_flag = 'fwd'
            else:
                commit_flag = ''
                break
        else:
            open(settings.logfile, 'a').write('\nAn incorrect commitor definition \"' + settings.commit_define_fwd[3][i] + '\" was given'
                                      ' for index ' + str(i) + ' in the \'fwd\' direction.')
            sys.exit('An incorrect commitor definition \"' + settings.commit_define_fwd[3][i] + '\" was given for index ' +
                     str(i) + ' in the \'fwd\' direction.')

    if commit_flag == '':                   # only bother checking for bwd commitment if not fwd commited
        for i in range(0, len(settings.commit_define_bwd[2])):
            if settings.commit_define_bwd[3][i] == 'lt':
                if pytraj.distance(traj, settings.commit_define_bwd[0][i] + ' ' + settings.commit_define_bwd[1][i], n_frames=1)[-1] <= \
                        settings.commit_define_bwd[2][i]:
                    commit_flag = 'bwd'     # if a committor test is passed, testing moves on to the next one.
                else:
                    commit_flag = ''
                    break                   # if a committor test is not passed, all testing in this direction ends in a fail
            elif settings.commit_define_bwd[3][i] == 'gt':
                if pytraj.distance(traj, settings.commit_define_bwd[0][i] + ' ' + settings.commit_define_bwd[1][i], n_frames=1)[-1] >= \
                        settings.commit_define_bwd[2][i]:
                    commit_flag = 'bwd'
                else:
                    commit_flag = ''
                    break
            else:
                open(settings.logfile, 'a').write('\nAn incorrect commitor definition \"' + settings.commit_define_bwd[3][i] + '\" was '
                                          'given for index ' + str(i) + ' in the \'bwd\' direction.')
                sys.exit('An incorrect commitor definition \"' + settings.commit_define_bwd[3][i] + '\" was given for index ' +
                         str(i) + ' in the \'bwd\' direction.')

    del traj  # to ensure cleanup of iterload object

    if commit_flag == '':
        commit_flag = 'fail'

    return commit_flag


def pickframe(thread, direction, forked_from=Thread(), frame=-1, suffix=1, settings=argparse.Namespace()):
    """
    Write a new restart-format coordinate file using a randomly chosen frame with index between 0 and settings.n_adjust
    to serve as the initial coordinates for a new simulation.

    This function loads the last valid trajectory from thread in the given direction in order to select a random frame.
    If forked_from is provided, it indicates that thread should be treated as a new Thread object but that its initial
    coordinates should be chosen from the forked_from Thread.

    If a frame argument >= 0 is given, this function does not use settings.n_adjust; instead, it simply uses that frame.
    If frame argument < 0 (which is the default), but settings.n_adjust is also negative, then the absolute value of
    that number is used (e.g., if settings.n_adjust = -10, the 10th frame is used).

    Parameters
    ----------
    thread : Thread or str
        The Thread object whose most recent simulation is to be checked for commitment. Can also be a string giving the
        name attribute of the trajectory to pick from. If this setting is a string, forked_from should always be used.
    direction : str
        Either 'fwd', 'bwd', or when forking, 'init', to indicate which simulation to pick the random frame from.
    forked_from : Thread
        A Thread object from which to draw the initial coordinates for thread. Default = Thread() (an empty Thread
        object, which will not be used.)
    frame : int
        Manually override the random frame picking by providing the frame index to use. Must be >= 0 to be used.
        Default = -1
    suffix : int
        Setting only used when thread is a string. This value is incremented by one and appended to the name of the new
        coordinate file. Default = 1
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    str
        Filename of the newly created restart-format coordinate file.

    """
    if not isinstance(thread,Thread):
        basename = thread   # thread should be a string
        last_valid = 1      # only used to avoid the if last_valid == '0' branch if forked_from is provided
    else:
        basename = thread.basename
        last_valid = thread.last_valid
        suffix = thread.suffix

    if last_valid == '0':           # to catch case where 'last_valid' is still the initial shooting point because a new valid transition path has not been found
        return thread.start_name    # will cause the new start_name to be unchanged from the previous run

    if frame >= 0:
        frame_number = frame
    elif settings.n_adjust < 0:
        frame_number = int(-1*settings.n_adjust)
    else:
        frame_number = random.randint(1, settings.n_adjust)

    if forked_from.basename:    # if an actual thread was given for forked_from (as opposed to the default)...
        if direction in ['fwd','bwd']:
            traj = pytraj.iterload(forked_from.basename + '_' + forked_from.last_valid + '_' + direction + '.nc', forked_from.prmtop, format='.nc',frame_slice=(frame_number,frame_number+1))
        elif direction == 'init':
            traj = pytraj.iterload(forked_from.basename + '_' + forked_from.last_valid + '_init_fwd.rst', forked_from.prmtop, format='.rst7')
        else:
            sys.exit('Error: pickframe() encountered unexpected direction: ' + direction + ' during fork')
        new_suffix = str(suffix)
    else:
        traj = pytraj.iterload(basename + '_' + last_valid + '_' + direction + '.nc', settings.topology, format='.nc',frame_slice=(frame_number,frame_number+1))
        new_suffix = str(int(suffix) + 1)

    new_restart_name = basename + '_' + new_suffix + '.rst7'
    pytraj.write_traj(new_restart_name,traj,options='multi',overwrite=True)    # multi because not including this option seems to keep it on anyway, so I want to be consistent
    try:
        os.rename(new_restart_name + '.1',new_restart_name)     # I don't quite know why, but pytraj appends '.1' to the filename, so this removes it.
    except OSError: # I sort of anticipate this breaking down the line, so this block is here to help handle that.
        open(settings.logfile, 'a').write('\nWarning: tried renaming .rst7.1 file ' + new_restart_name + '.1 with .rst7'
                                  ', but encountered OSError exception. Either you ran out of storage space, or this is'
                                  ' a possible indication of an unexpected pytraj version?')
        open(settings.logfile, 'a').close()
        if not os.path.exists(new_restart_name):
            open(settings.logfile, 'a').write('\nError: pickframe did not produce the restart file for the next shooting move. '
                                      'Please ensure that you didn\'t run out of storage space, and then raise this '
                                      'issue on GitHub to let me know!')
            sys.exit('Error: pickframe did not produce the restart file for the next shooting move. Please ensure that '
                     'you didn\'t run out of storage space, and then raise this issue on GitHub to let me know!')
        else:
            open(settings.logfile, 'a').write('\nWarning: it tentatively looks like this should be okay, as the desired file was still created.')
            open(settings.logfile, 'a').close()
        pass

    if forked_from.basename:
        open(settings.logfile, 'a').write('\nForking ' + basename + ' from ' + forked_from.basename + '_' + forked_from.last_valid + '_' + direction + '.nc, frame number ' + str(frame_number))
    else:
        open(settings.logfile, 'a').write('\nInitializing next shooting point from shooting run ' + basename + '_' + last_valid + ' in ' + direction + ' direction, frame number ' + str(frame_number))
    open(settings.logfile, 'a').close()

    del traj  # to ensure cleanup of iterload object

    return new_restart_name


def get_rc_value(cv_values, settings):
    """
    Evaluate and return reaction coordinate value based on a given list of cv_values.

    Parameters
    ----------
    cv_values : list
        A list of values for the CVs to plug into the RC equation as needed (same order as candidateops)
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    None
    """
    equation = settings.rc_definition
    if settings.literal_ops:
        local_candidateops = [settings.candidateops]  # to fix error where candidateops has unexpected format
    else:
        local_candidateops = settings.candidateops
    if settings.include_qdot:
        qdot_factor = 2  # to include qdot CVs if applicable
    else:
        qdot_factor = 1
    for j in reversed(range(int(qdot_factor * len(local_candidateops[0])))):  # for each candidate cv... (reversed to avoid e.g. 'CV10' -> 'cv_values[0]0')
        equation = equation.replace('CV' + str(j + 1), 'cv_values[' + str(j) + ']')
    return eval(str(equation))


def cleanthread(thread, settings):
    """
    Reset thread parameters in preparation for the next step of aimless shooting after the previous one has completed.
    Add the next step to the itinerary if appropriate. Also write to history and output files, implement fork if
    necessary, and terminate the thread if any of the termination criteria are met, among other housekeeping tasks.

    This function should be called after every thread step is completed to handle it in the appropriate manner. In
    effect, it serves as a housekeeping function to take care of all the important details that are checked for after
    every "prod" step.

    Parameters
    ----------
    thread : Thread
        The Thread object that just completed a move.
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    None

    """
    # global candidateops

    def report_rc_values(coord_file):
        # Simple function for outputting the RC values for a given trajectory traj to the eps_results.out file
        # todo: replace use of traj with simple evaluation of eps_fwd/bwd variable, depending on direction argument? (Unimportant, optimization only)
        rc_values = []
        if '.rst' in coord_file or '.rst7' in coord_file:
            fileformat = '.rst7'
        elif '.nc' in coord_file:
            fileformat = '.nc'
        else:
            sys.exit('Error: cleanthread.report_rc_values() encountered a file of unknown format: ' + coord_file)
        traj = pytraj.iterload(coord_file, thread.prmtop, format=fileformat)
        for i in range(traj.__len__()):                         # iterate through frames of traj
            cv_values = [float(cv) for cv in candidatevalues(coord_file, frame=i, reduce=True, settings=settings).split(' ') if cv]  # CV values as a list
            rc_values.append(get_rc_value(cv_values=cv_values, settings=settings))
        for value in rc_values:
            if thread.rc_min <= value <= thread.rc_max:     # only write to output if the bead is inside the window
                open('eps_results.out', 'a').write(str(thread.rc_min) + ' ' + str(thread.rc_max) + ' ' + str(value) + '\n')
                open('eps_results.out', 'a').close()
        return rc_values

    if settings.eps_settings:                # EPS behavior
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
        if settings.eps_dynamic_seed and (True in [bool(x) for x in settings.empty_windows]) and (thread.last_valid == thread.suffix): # is this last boolean required? I think maybe yes because I'm using pickframe()?
            rc_values = list(reversed(bwd_rc_values)) + init_rc_value + fwd_rc_values
            start_bead = 0
            suffix = 1
            for rc_value in rc_values:
                start_bead += 1
                for window in range(len(settings.eps_windows) - 1):
                    if (settings.empty_windows[window] > 0) and (settings.eps_windows[window] - settings.overlap <= rc_value <= settings.eps_windows[window + 1] + settings.overlap):
                        # Write a new coordinate file from the appropriate trajectory
                        # todo: this is so ugly because I didn't design pickframe() to help make a new thread with an unknown initial structure. Can I clean this up somehow?
                        if start_bead <= thread.eps_bwd_la:         # use la values since pickframe uses the la trajectory
                            structure = pickframe(thread.name, 'bwd', frame=int(thread.eps_bwd_la - start_bead), forked_from=thread, suffix=suffix, settings=settings)     # "frame" should be zero-indexed
                            suffix += 1
                            debug_dir = 'bwd'
                            debug_frame = int(thread.eps_bwd_la - start_bead)
                        elif start_bead == thread.eps_bwd_la + 1:   # the initial coordinates
                            structure = pickframe(thread.name, 'init', forked_from=thread, suffix=suffix, settings=settings)
                            suffix += 1
                            debug_dir = 'init_fwd'
                            debug_frame = 'N/A'
                        else:                                       # inside the fwd trajectory
                            structure = pickframe(thread.name, 'fwd', frame=int(start_bead - thread.eps_bwd_la - 2), forked_from=thread, suffix=suffix, settings=settings)   # "frame" should be zero-indexed
                            suffix += 1
                            debug_dir = 'fwd'
                            debug_frame = int(start_bead - thread.eps_bwd_la - 1)
                        settings.dynamic_seed_kludge = window   # forces spawnthread to place this thread in the correct window in the case where it could fit into two due to overlap
                        newthread = spawnthread(structure, suffix='1', settings=settings)  # spawn a new thread with the default settings
                        #newthread.last_valid = '0'              # so that if the first shooting point does not result in a valid transition path, shooting will begin from the TS guess
                        newthread.prmtop = settings.topology    # set prmtop filename for the thread
                        settings.itinerary.append(newthread)    # submit it to the itinerary
                        open(settings.logfile, 'a').write('\nEmpty EPS window with upper and lower boundaries: ' +
                                                  str(settings.eps_windows[window] - settings.overlap) + ' and ' +
                                                  str(settings.eps_windows[window + 1] + settings.overlap) + ' has been'
                                                  ' seeded using bead ' + str(start_bead) + ' from shooting move ' +
                                                  thread.name + '. Debug information:')
                        open(settings.logfile, 'a').write('\n  fwd_rc_values = ' + str(fwd_rc_values))
                        open(settings.logfile, 'a').write('\n  bwd_rc_values = ' + str(bwd_rc_values))
                        open(settings.logfile, 'a').write('\n  rc_values = ' + str(rc_values))
                        open(settings.logfile, 'a').write('\n  start_bead = ' + str(start_bead))
                        open(settings.logfile, 'a').write('\n  pickframe trajectory = ' + thread.basename + '_' + thread.last_valid + '_' + debug_dir + '.nc')
                        open(settings.logfile, 'a').write('\n  frame from trajectory = ' + str(debug_frame))
                        open(settings.logfile, 'a').write('\n  structure = ' + str(structure))
                        open(settings.logfile, 'a').write('\n  new empty_windows = ' + str(settings.empty_windows))
                        open(settings.logfile, 'a').close()

    elif thread.commit1 != 'fail':  # standard aimless shooting behavior
        # Record result of forward trajectory in output file. This is done regardless of whether the shooting point was
        # accepted; accept/reject is for keeping the sampling around the separatrix, but even rejected points are valid
        # for calculating the reaction coordinate so long as they committed to a basin!
        if thread.commit1 == 'fwd':
            basin = 'B'
        elif thread.commit1 == 'bwd':
            basin = 'A'
        else:
            basin = thread.commit1
            sys.exit('Error: thread commit1 flag took on unexpected value: ' + basin + '\nThis is a weird error.'
                     ' Please raise this issue on GitHub along with your ATESA input file!')
        open('as.out', 'a').write(basin + ' <- ' + candidatevalues(thread.name + '_init_fwd.rst', settings=settings) + '\n')
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
    try:
        with open('history/' + thread.basename, 'w') as file:
            for history_line in thread.history:
                file.write(history_line + '\n')
            file.close()
    except (IOError, OSError):
        if os.path.exists(settings.working_directory + '/history'):
            shutil.rmtree(settings.working_directory + '/history')   # delete old (apparently broken) history directory
            os.makedirs(settings.working_directory + '/history')     # make a new one
        else:
            os.makedirs(settings.working_directory + '/history')
        with open('history/' + thread.basename, 'w') as file:
            for history_line in thread.history:
                file.write(history_line + '\n')
            file.close()

    thread.total_moves += 1
    open(settings.logfile, 'a').write('\nShooting run ' + thread.name + ' finished with fwd trajectory result: ' + thread.commit1 + ' and bwd trajectory result: ' + thread.commit2)
    if settings.eps_settings:
        open(settings.logfile, 'a').write(', as well as init result: ' + checkcommit(thread, 'init', settings=settings))   # todo: should probably save an init_commit attribute to threads to avoid checking commitment on init for a second time here.
    open(settings.logfile, 'a').write('\n' + thread.basename + ' has a current acceptance ratio of: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or ' + str(100*thread.accept_moves/thread.total_moves)[0:5] + '%')
    open(settings.logfile, 'a').close()

    # Implementation of fork. Makes (fork - 1) new threads from successful runs and adds them to the itinerary. The new
    # threads do not inherit anything from their parents except starting point and history.
    if settings.fork > 1 and thread.last_valid == thread.suffix:
        for i in range(settings.fork - 1):
            direction = random.randint(0, 1)
            if direction == 0:
                pick_dir = 'fwd'
            else:
                pick_dir = 'bwd'
            newthread = spawnthread(thread.name + '_' + str(i + 1), suffix='1', settings=settings)
            newthread.prmtop = thread.prmtop
            newthread.start_name = pickframe(newthread, pick_dir, thread, settings=settings)
            #newthread.last_valid = '0'
            newthread.history = thread.history
            settings.itinerary.append(newthread)

    if settings.eps_settings:    # EPS behavior
        start_bead = random.randint(1, settings.k_beads)
        # Thread has attributes eps_fwd and eps_bwd telling me how long the fwd and bwd trajectories are...
        if start_bead <= thread.eps_bwd_la:         # use la values since pickframe uses the la trajectory
            thread.start_name = pickframe(thread, 'bwd', frame=int(thread.eps_bwd_la - start_bead), settings=settings)         # "frame" should be zero-indexed
        elif start_bead == thread.eps_bwd_la + 1:   # the initial coordinates
            thread.start_name = thread.name + '_init_fwd.rst'
        else:                                       # inside the fwd trajectory
            thread.start_name = pickframe(thread, 'fwd', frame=int(start_bead - thread.eps_bwd_la - 2), settings=settings)     # "frame" should be zero-indexed
        thread.eps_fwd = settings.k_beads - start_bead           # set new eps_fwd and _bwd to keep string length the same
        thread.eps_bwd = settings.k_beads - thread.eps_fwd - 1   # extra -1 to account for starting point
        if settings.cleanup:
            if not thread.suffix == thread.last_valid:
                if os.path.exists(thread.basename + '_' + thread.suffix + '_fwd.nc'):
                    os.remove(thread.basename + '_' + thread.suffix + '_fwd.nc')
                if os.path.exists(thread.basename + '_' + thread.suffix + '_bwd.nc'):
                    os.remove(thread.basename + '_' + thread.suffix + '_bwd.nc')
    else:               # normal aimless shooting behavior
        direction = random.randint(0, 1)  # This is necessary to avoid an issue where acceptance ratios fall off as sampling progresses. See Mullen et al. 2015 (Easy TPS) SI.
        if direction == 0:
            pick_dir = 'fwd'
        else:
            pick_dir = 'bwd'
        if thread.last_valid == thread.suffix or settings.always_new:  # pick a new starting point if the last move was a success
            thread.start_name = pickframe(thread, pick_dir, settings=settings)

    if settings.cleanup:    # init trajectory is never used for anything, so delete it if settings.cleanup == True
        if os.path.exists(thread.basename + '_' + thread.suffix + '_init.nc'):
            os.remove(thread.basename + '_' + thread.suffix + '_init.nc')

    thread.type = 'init'
    thread.suffix = str(int(thread.suffix) + 1)
    thread.name = thread.basename + '_' + thread.suffix
    thread.jobid1 = ''  # this line required if eps_settings is given, redundant otherwise
    thread.jobid2 = ''  # this line required if eps_settings is given, redundant otherwise
    thread.commit1 = ''
    thread.commit2 = ''

    try:
        null = settings.bootstrap_bookkeep  # to throw AttributeError if not set up to do bootstrapping
        settings.bootstrap_flag = handle_bootstrap(settings)  # handles tasks associated with bootstrapping
        if settings.bootstrap_flag == True:
            open(settings.logfile, 'a').write('\nBootstrapped reaction coordinates agree to within given tolerance. No '
                                              'further jobs will be submitted by this instance of ATESA, but currently'
                                              'running jobs will be allowed to finish. To perform more sampling, submit'
                                              ' a new ATESA job in the same working directory with restart = True')
            open(settings.logfile, 'a').close()
    except AttributeError:
        pass

    if thread.failcount >= settings.max_fails > 0:
        thread.status = 'max_fails'     # the thread dies because it has failed too many times in a row
    elif thread.total_moves >= settings.max_moves > 0:
        thread.status = 'max_moves'     # the thread dies because it has performed too many total moves
    elif thread.accept_moves >= settings.max_accept > 0:
        thread.status = 'max_accept'    # the thread dies because it has accepted too many moves
    else:
        try:
            if not settings.bootstrap_flag:
                settings.itinerary.append(thread)        # the thread lives and moves to next step
        except AttributeError:
            settings.itinerary.append(thread)  # the thread lives and moves to next step

    # Write a status file to indicate the acceptance ratio and current status of every thread.
    with open('status.txt','w') as file:
        for thread in settings.allthreads:
            try:
                file.write(thread.basename + ' acceptance ratio: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or ' + str(100*thread.accept_moves/thread.total_moves)[0:5] + '%\n')
            except ZeroDivisionError:   # Since any thread that hasn't completed a move yet has total_moves = 0
                file.write(thread.basename + ' acceptance ratio: ' + str(thread.accept_moves) + '/' + str(thread.total_moves) + ', or 0%\n')
            if thread in settings.itinerary:
                file.write('  Status: move ' + thread.suffix + ' queued\n')
            elif thread in settings.running:
                file.write('  Status: move ' + thread.suffix + ' running\n')
            else:
                if thread.status in ['max_accept','max_moves','max_fails']:
                    file.write('  Status: terminated after move ' + thread.suffix + ' due to termination criterion: ' + thread.status + '\n')
                else:
                    file.write('  Status: crashed during move ' + thread.suffix + '\n')
        file.close()


def candidatevalues(coord_file, frame=-1, reduce=False, settings=argparse.Namespace()):
    """
    Evaluate the candidate CV values from the initial coordinates of the most recent step of the given thread.

    This function is only capable of returning order parameter rate of change values when the literal_ops Boolean is
    True. This will change in future versions.

    Parameters
    ----------
    coord_file : str
        The name of the coordinate file to evaluate.
    frame : int
        Frame number to evaluate instead of the most recent, zero-indexed. Must be >= 0 to be used.
    reduce : bool
        Boolean indicating whether or not to reduce the candidate OP values based on the contents of rc_minmax. This
        should be used when the output of candidatevalues is being used to calculate an RC value.
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    str
        A space-separated series of CV values in the same order as they are given in candidate_cv, followed
        by their rates of change per 1/20.455 ps (unit of time in Amber) in the same order if applicable.

    """
    def reduce_op(input,local_index):
        # Return the reduced value of the index'th OP with un-reduced value input (an int or float).
        try:
            if not isinstance(float(settings.rc_minmax[0][local_index]),float) or not isinstance(float(settings.rc_minmax[1][local_index]),float):  # if there's a blank entry in rc_minmax
                sys.exit('Error: rc_definition contains reference to OP' + str(local_index+1) + ' without a corresponding entry in rc_minmax')
        except (IndexError, ValueError):                         # if there's no entry at all
            sys.exit('Error: rc_definition contains reference to OP' + str(local_index + 1) + ' without a corresponding entry in rc_minmax')
        return (input - settings.rc_minmax[0][local_index])/(settings.rc_minmax[1][local_index] - settings.rc_minmax[0][local_index])

    if '.rst' in coord_file or '.rst7' in coord_file:
        fileformat = '.rst7'
    elif '.nc' in coord_file:
        fileformat = '.nc'
    else:
        sys.exit('Error: candidatevalues() encountered a file of unsupported file extension (only .rst, .rst7, and .nc '
                 'are supported): ' + coord_file)

    output = ''
    if frame >= 0:
        traj = pytraj.iterload(coord_file, settings.topology, format=fileformat, frame_slice=(frame,frame+1))
        if settings.include_qdot:
            pytraj.write_traj('temp_frame.rst7',traj,overwrite=True,velocity=True,options='multi')    # multi because not including this option seems to keep it on anyway, so I want to be consistent
            try:
                os.rename('temp_frame.rst7.1','temp_frame.rst7')    # I don't quite know why, but pytraj appends '.1' to the filename, so this removes it.
            except OSError:                                         # I sort of anticipate this breaking down the line, so this block is here to help handle that.
                open(settings.logfile, 'a').write('\nWarning: tried renaming .rst7.1 file temp_frame.rst7.1 with .rst7,'
                                          ' but encountered OSError exception. Either you ran out of storage space, or '
                                          'this is a possible indication of an unexpected pytraj version?')
                open(settings.logfile, 'a').close()
                if not os.path.exists('temp_frame.rst7'):
                    sys.exit('Error: pickframe did not produce the restart file for the next shooting move. Please ensure that you didn\'t run out of storage space, and then raise this issue on GitHub to let me know!')
                else:
                    open(settings.logfile, 'a').write('\nWarning: it tentatively looks like this should be okay, as the desired file was still created.')
                    open(settings.logfile, 'a').close()
                pass
    else:
        traj = pytraj.iterload(coord_file, settings.topology, format=fileformat)

    def increment_coords():
        # Modified from revvels() to increment coordinate values by velocities, rather than reversing velocities.
        # Returns the name of the newly-created coordinate file
        if not frame >= 0:
            filename = coord_file
        else:
            filename = 'temp_frame.rst7'
        byline = open(filename).readlines()
        pattern = re.compile('-*[0-9.]+')           # regex to match numbers including decimals and negatives
        n_atoms = pattern.findall(byline[1])[0]     # number of atoms indicated on second line of .rst file

        shutil.copyfile(filename, 'temp.rst')
        for i, line in enumerate(fileinput.input('temp.rst', inplace=1)):
            if int(n_atoms)/2 + 2 > i >= 2:
                newline = line
                coords = pattern.findall(newline)                                          # line of coordinates
                try:
                    vels = pattern.findall(byline[i + int(math.ceil(int(n_atoms)/2))])     # corresponding velocities
                except IndexError:
                    sys.exit('Error: candidatevalues.increment_coords() encountered an IndexError. This is caused '
                             'by attempting to read qdot values from a coordinate file lacking velocity information, or'
                             ' else by that file being truncated. The offending file is: ' + filename)
                # Sometimes items in coords or vels 'stick together' at a negative sign (e.g., '-1.8091748-112.6420521')
                # This next loop is just to split them up
                for index in range(len(coords)):
                    length = len(coords[index])                     # length of string representing this coordinate
                    replace_string = str(float(coords[index]) + float(vels[index]))[0:length-1]
                    while len(replace_string) < length:
                        replace_string += '0'
                    newline = newline.replace(coords[index], replace_string)
                sys.stdout.write(newline)
            else:
                sys.stdout.write(line)

        return 'temp.rst'

    # Implementation of explicit, to directly interpret user-supplied OPs
    if settings.literal_ops:
        values = []
        local_index = -1
        for op in settings.candidateops:
            local_index += 1
            evaluation = eval(op)
            if settings.include_qdot:    # want to save values for later
                values.append(float(evaluation))
            if reduce:
                evaluation = reduce_op(evaluation, local_index)
            output += str(evaluation) + ' '
        if settings.include_qdot:        # if True, then we want to include rate of change for every OP, too
            # Strategy here is to write a new temporary .rst7 file by incrementing all the coordinate values by their
            # corresponding velocity values, load it as a new iterload object, and then rerun our analysis on that.
            traj = pytraj.iterload(increment_coords(), settings.topology)
            local_index = -1
            for op in settings.candidateops:
                local_index += 1
                evaluation = eval(op) - values[local_index]     # Subtract value 1/20.455 ps earlier from value of op
                if reduce:
                    evaluation = reduce_op(evaluation,local_index + len(settings.candidateops))
                output += str(evaluation) + ' '

    # todo: test implementation of include_qdot for this branch of candidatevalues
    else:
        values = []
        for local_index in range(0,len(settings.candidateops[0])):
            if len(settings.candidateops) == 4:          # settings.candidateops contains dihedrals
                if settings.candidateops[3][local_index]:      # if this OP is a dihedral
                    value = pytraj.dihedral(traj,mask=settings.candidateops[0][local_index] + ' ' + settings.candidateops[1][local_index] + ' ' + settings.candidateops[2][local_index] + ' ' + settings.candidateops[3][local_index])[0]
                elif settings.candidateops[2][local_index]:    # if this OP is an angle
                    value = pytraj.angle(traj,mask=settings.candidateops[0][local_index] + ' ' + settings.candidateops[1][local_index] + ' ' + settings.candidateops[2][local_index])[0]
                else:                           # if this OP is a distance
                    value = pytraj.distance(traj,mask=settings.candidateops[0][local_index] + ' ' + settings.candidateops[1][local_index])[0]
            elif len(settings.candidateops) == 3:        # settings.candidateops contains angles but not dihedrals
                if settings.candidateops[2][local_index]:      # if this OP is an angle
                    value = pytraj.angle(traj,mask=settings.candidateops[0][local_index] + ' ' + settings.candidateops[1][local_index] + ' ' + settings.candidateops[2][local_index])[0]
                else:                           # if this OP is a distance
                    value = pytraj.distance(traj,mask=settings.candidateops[0][local_index] + ' ' + settings.candidateops[1][local_index])[0]
            else:                               # settings.candidateops contains only distances
                value = pytraj.distance(traj,mask=settings.candidateops[0][local_index] + ' ' + settings.candidateops[1][local_index])[0]

            if settings.include_qdot:    # want to save values for later
                values.append(float(value))
            if reduce:
                value = reduce_op(value, local_index)
            output += str(value) + ' '
        if settings.include_qdot:        # if True, then we want to include rate of change for every OP, too
            # Strategy here is to write a new temporary .rst7 file by incrementing all the coordinate values by their
            # corresponding velocity values, load it as a new iterload object, and then rerun our analysis on that.
            traj = pytraj.iterload(increment_coords(), settings.topology)
            for local_index in range(0, len(settings.candidateops[0])):
                if len(settings.candidateops) == 4:  # settings.candidateops contains dihedrals
                    if settings.candidateops[3][local_index]:  # if this OP is a dihedral
                        value = pytraj.dihedral(traj, mask=settings.candidateops[0][local_index] + ' ' +
                                                           settings.candidateops[1][local_index] + ' ' +
                                                           settings.candidateops[2][local_index] + ' ' +
                                                           settings.candidateops[3][local_index])[0]
                    elif settings.candidateops[2][local_index]:  # if this OP is an angle
                        value = pytraj.angle(traj, mask=settings.candidateops[0][local_index] + ' ' +
                                                        settings.candidateops[1][local_index] + ' ' +
                                                        settings.candidateops[2][local_index])[0]
                    else:  # if this OP is a distance
                        value = pytraj.distance(traj, mask=settings.candidateops[0][local_index] + ' ' +
                                                           settings.candidateops[1][local_index])[0]
                elif len(settings.candidateops) == 3:  # settings.candidateops contains angles but not dihedrals
                    if settings.candidateops[2][local_index]:  # if this OP is an angle
                        value = pytraj.angle(traj, mask=settings.candidateops[0][local_index] + ' ' +
                                                        settings.candidateops[1][local_index] + ' ' +
                                                        settings.candidateops[2][local_index])[0]
                    else:  # if this OP is a distance
                        value = pytraj.distance(traj, mask=settings.candidateops[0][local_index] + ' ' +
                                                           settings.candidateops[1][local_index])[0]
                else:  # settings.candidateops contains only distances
                    value = pytraj.distance(traj, mask=settings.candidateops[0][local_index] + ' ' +
                                                       settings.candidateops[1][local_index])[0]
                value = value - values[local_index]     # Subtract value 1/20.455 ps earlier from value of op
                if reduce:
                    value = reduce_op(value, local_index + len(settings.candidateops[0]))
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
def main_loop(settings):
    """
    Perform the sequence of simulations constituting a full ATESA run.

    This is the primary runtime loop of this program, from which all the helper functions above are called (either
    directly or by one another). Along with the helper functions, it builds Threads, submits them to the batch system,
    monitors them for completion, and handles the output and next steps once they've finished.

    Parameters
    ----------
    settings : Namespace
        Contains all the settings dictating how this function operates.

    Returns
    -------
    None

    """
    # These bash commands will be used later to cancel jobs after they've completed
    if settings.batch_system == 'pbs':
        cancel_command = 'qdel'
    elif settings.batch_system == 'slurm':
        cancel_command = 'scancel'

    # global itinerary                # get global variables from outer scope so they are accessible by main_loop()
    # global running                  # honestly I don't know why this is necessary, but removing these lines breaks it
    # global allthreads

    while settings.itinerary or settings.running:     # while either list has contents...
        itin_names = [thread.name + '_' + thread.type for thread in settings.itinerary]
        run_names = [thread.name + '_' + thread.type for thread in settings.running]
        open(settings.logfile, 'a').write(
            '\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))
        open(settings.logfile, 'a').write('\nSubmitting jobs in itinerary...')
        open(settings.logfile, 'a').close()

        local_index = -1  # set place in itinerary to -1
        for thread in settings.itinerary:   # for each thread that's ready for its next step...
            local_index += 1                # increment place in itinerary
            makebatch(thread,settings)      # make the necessary batch file
            if thread.type == 'init':
                thread.jobid1 = subbatch(thread, 'init', settings=settings)    # submit batch file and collect jobID into thread
            else:
                try:
                    if not thread.eps_fwd == 0:     # if this setting exists and isn't zero
                        thread.jobid1 = subbatch(thread, 'fwd', settings=settings)
                    else:
                        thread.commit1 = 'False'    # simulation doesn't run, so it doesn't satisfy acceptance criterion
                except:                             # if this setting doesn't exist (normal aimless shooting behavior)
                    thread.jobid1 = subbatch(thread, 'fwd', settings=settings)
                try:                                # repeat for bwd half of thread
                    if not thread.eps_bwd == 0:
                        thread.jobid2 = subbatch(thread, 'bwd', settings=settings)
                    else:
                        thread.commit2 = 'False'
                except:
                    thread.jobid2 = subbatch(thread, 'bwd', settings=settings)
            settings.running.append(settings.itinerary[local_index])    # move itinerary element to next in running

        settings.itinerary = []  # empty itinerary

        itin_names = [thread.name + '_' + thread.type for thread in settings.itinerary]
        run_names = [thread.name + '_' + thread.type for thread in settings.running]
        open(settings.logfile, 'a').write('\nCurrent status...\n Itinerary: ' + str(itin_names) + '\n Running: ' + str(run_names))
        open(settings.logfile, 'a').close()

        # Once again, the fact that groupfile support was only added late in development means that this is a bit ugly.
        # This while loop serves the same purpose as the one directly following it, but works in terms of groupfile jobs
        # instead of individual batch jobs. I can't think of an easy way to unify them, but this is an opportunity for
        # polish if I ever rewrite this software from the ground up.
        #
        # I'm not implementing flexible-length shooting for groupfiles, since it should be exceedingly rare anyway for
        # every job in a groupfile to be committed substantially before the job is ending anyway, assuming the user
        # chooses prod_walltime judiciously.
        # todo: implement EPS for groupfile > 0
        while settings.groupfile > 0 and not settings.itinerary:
            handle_groupfile()                          # update settings.groupfile_list
            for groupfile_data in settings.groupfile_list:       # for each list element [<NAME>, <STATUS>, <CONTENTS>, <TIME>]
                if groupfile_data[1] == 'completed':    # if this groupfile task has status "completed"
                    for string in groupfile_data[2].split(' '):     # iterate through thread.name + '_' + thread.type of jobs in groupfile
                        if 'init' in string:                        # do this for all init jobs in this groupfile
                            # Need to get ahold of the thread this init job represents...
                            for thread in settings.running:       # can't say this is surely the best way to do this...
                                if thread.name in string:
                                    try:
                                        revvels(thread)     # make the initial .rst file for the bwd trajectory
                                        settings.itinerary.append(thread)
                                        thread.type = 'prod'
                                        open(settings.logfile, 'a').write('\nJob completed: ' + thread.name + '_init\nAdding ' + thread.name + ' forward and backward jobs to itinerary')
                                        open(settings.logfile, 'a').close()
                                    except (IOError, OSError):   # when revvels can't find the init .rst file
                                        open(settings.logfile, 'a').write('\nThread ' + thread.basename + ' crashed: initialization did not produce a restart file.')
                                        if not settings.restart_on_crash:
                                            open(settings.logfile, 'a').write('\nrestart_on_crash = False; thread will not restart')
                                            open(settings.logfile, 'a').close()
                                        elif not settings.restart_on_crash:
                                            open(settings.logfile, 'a').write('\nrestart_on_crash = True; resubmitting thread to itinerary')
                                            open(settings.logfile, 'a').close()
                                            settings.itinerary.append(thread)
                                            thread.type = 'init'
                                            settings.running.remove(thread)
                        if 'prod' in string:                        # do this for all prod jobs in this groupfile
                            # todo: figure out how to tolerate only one of the halves of a prod job showing up in a given groupfile
                            # Need to get ahold of the thread this prod job represents...
                            for thread in settings.running:       # can't say this is surely the best way to do this...
                                if thread.name in string:
                                    thread.commit1 = checkcommit(thread, 'fwd', settings=settings)  # check for commitment in fwd job
                                    if not thread.commit1:
                                        thread.commit1 = 'fail'
                                    thread.commit2 = checkcommit(thread, 'bwd', settings=settings)  # check for commitment in bwd job
                                    if not thread.commit2:
                                        thread.commit2 = 'fail'
                                    settings.running.remove(thread)
                                    thread.failcount += 1       # increment fails in a row regardless of outcome
                                    if settings.eps_settings and thread.commit1 == 'True' or thread.commit2 == 'True' or (checkcommit(thread, 'init', settings=settings) is True):  # valid EPS move
                                        if settings.cleanup:
                                            if os.path.exists(thread.basename + '_' + thread.last_valid + '_fwd.nc'):
                                                os.remove(thread.basename + '_' + thread.last_valid + '_fwd.nc')
                                            if os.path.exists(thread.basename + '_' + thread.last_valid + '_bwd.nc'):
                                                os.remove(thread.basename + '_' + thread.last_valid + '_bwd.nc')
                                        thread.last_valid = thread.suffix
                                        thread.accept_moves += 1
                                        thread.failcount = 0    # reset fail count to zero if this move was accepted
                                    elif thread.commit1 != thread.commit2 and thread.commit1 != 'fail' and thread.commit2 != 'fail':  # valid transition path, update 'last_valid' attribute
                                        if settings.cleanup:
                                            if os.path.exists(thread.basename + '_' + thread.last_valid + '_fwd.nc'):
                                                os.remove(thread.basename + '_' + thread.last_valid + '_fwd.nc')
                                            if os.path.exists(thread.basename + '_' + thread.last_valid + '_bwd.nc'):
                                                os.remove(thread.basename + '_' + thread.last_valid + '_bwd.nc')
                                        thread.last_valid = thread.suffix
                                        thread.accept_moves += 1
                                        thread.failcount = 0    # reset fail count to zero if this move was accepted
                                    elif not settings.eps_settings and settings.cleanup:  # shooting move did not pass, so we can immediately delete the trajectories
                                        if os.path.exists(thread.basename + '_' + thread.suffix + '_fwd.nc'):
                                            os.remove(thread.basename + '_' + thread.suffix + '_fwd.nc')
                                        if os.path.exists(thread.basename + '_' + thread.suffix + '_bwd.nc'):
                                            os.remove(thread.basename + '_' + thread.suffix + '_bwd.nc')
                                    cleanthread(thread, settings=settings)
                    groupfile_data[1] = 'processed'     # to indicate that this completed groupfile job has been handled

            pickle.dump(settings.allthreads, open('restart.pkl', 'wb'), protocol=2)  # dump information required to restart ATESA
            if not settings.itinerary:
                time.sleep(60)                          # delay 60 seconds before checking for job status again

            if not settings.itinerary and not settings.running and not settings.eps_settings:
                acceptances = [(100 * thread.accept_moves / thread.total_moves) for thread in settings.allthreads]
                open(settings.logfile, 'a').write(
                    '\nItinerary and running lists are empty.\nAimless shooting is complete! '
                    'The highest acceptance ratio for any thread was ' + str(max(acceptances)) +
                    '%.\nSee as.out in the working directory for results.')
                open(settings.logfile, 'a').close()
                break
            elif not settings.itinerary and not settings.running and settings.eps_settings:
                open(settings.logfile, 'a').write(
                    '\nItinerary and running lists are empty.\nEquilibrium path sampling is complete!'
                    '\nSee eps_results.out in the working directory for results.')
                open(settings.logfile, 'a').close()
                break

        while settings.groupfile == 0 and not settings.itinerary:   # while itinerary is empty and not in groupfile mode
            output = interact('queue',settings)     # retrieves string containing jobids of running and queued jobs
            local_index = 0                         # set place in running to 0
            while local_index < len(settings.running):  # while instead of for to control indexing manually
                thread = settings.running[local_index]
                if thread.jobid1 and not thread.jobid1 in str(output):  # if a submitted job is no longer running
                    if thread.type == 'init':
                        try:
                            revvels(thread)  # make the initial .rst file for the bwd trajectory
                            settings.itinerary.append(settings.running[local_index])
                            thread.type = 'prod'
                            thread.type = 'prod'
                            open(settings.logfile, 'a').write(
                                '\nJob completed: ' + thread.name + '_init\nAdding ' + thread.name + ' forward and backward jobs to itinerary')
                            open(settings.logfile, 'a').close()
                        except (IOError, OSError):  # when revvels can't find the init .rst file
                            open(settings.logfile, 'a').write('\nThread ' + thread.basename + ' crashed: initialization did not produce a restart file.')
                            if not settings.restart_on_crash:
                                open(settings.logfile, 'a').write('\nrestart_on_crash = False; thread will not restart')
                                open(settings.logfile, 'a').close()
                            elif settings.restart_on_crash:
                                open(settings.logfile, 'a').write('\nrestart_on_crash = True; resubmitting thread to itinerary')
                                open(settings.logfile, 'a').close()
                                settings.itinerary.append(settings.running[local_index])
                                thread.type = 'init'
                        del settings.running[local_index]
                        local_index -= 1  # to keep index on track after deleting an entry
                    elif thread.type == 'prod':
                        # fwd trajectory exited before passing a commitor test, either walltime or other error
                        thread.commit1 = checkcommit(thread, 'fwd', settings=settings)  # check one last time
                        if not thread.commit1:
                            thread.commit1 = 'fail'
                        thread.jobid1 = ''
                if thread.jobid2 and not thread.jobid2 in str(output):  # if one of the submitted jobs no longer appears to be running
                    # bwd trajectory exited before passing a commitor test, either walltime or other error
                    thread.commit2 = checkcommit(thread, 'bwd', settings=settings)  # check one last time
                    if not thread.commit2:
                        thread.commit2 = 'fail'
                    thread.jobid2 = ''
                local_index += 1  # increment place in running

            local_index = 0
            while local_index < len(settings.running):
                thread = settings.running[local_index]
                if thread.type == 'prod':
                    if not thread.commit1 and not settings.eps_settings:     # 'and not eps_settings' ensures EPS jobs only get checked for 'commitment' when they exit naturally.
                        thread.commit1 = checkcommit(thread, 'fwd', settings=settings)
                    if not thread.commit2 and not settings.eps_settings:
                        thread.commit2 = checkcommit(thread, 'bwd', settings=settings)
                    if thread.commit1 and thread.jobid1 and not settings.eps_settings:
                        process = subprocess.Popen([cancel_command, thread.jobid1], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()  # doesn't do anything, I think
                        thread.jobid1 = ''
                    if thread.commit2 and thread.jobid2 and not settings.eps_settings:
                        process = subprocess.Popen([cancel_command, thread.jobid2], stdout=subprocess.PIPE)
                        (output, err) = process.communicate()  # doesn't do anything, I think
                        thread.jobid2 = ''
                    if thread.commit1 and thread.commit2:
                        del settings.running[local_index]
                        local_index -= 1  # to keep index on track after deleting an entry
                        thread.failcount += 1  # increment fails in a row regardless of outcome
                        if settings.eps_settings and thread.commit1 == 'True' or thread.commit2 == 'True' or (checkcommit(thread, 'init', settings=settings) is True):   # valid EPS move
                            if not (thread.commit1 == 'fail' or thread.commit2 == 'fail'):
                                if settings.cleanup:
                                    if os.path.exists(thread.basename + '_' + thread.last_valid + '_fwd.nc'):
                                        os.remove(thread.basename + '_' + thread.last_valid + '_fwd.nc')
                                    if os.path.exists(thread.basename + '_' + thread.last_valid + '_bwd.nc'):
                                        os.remove(thread.basename + '_' + thread.last_valid + '_bwd.nc')
                                thread.last_valid = thread.suffix
                                thread.accept_moves += 1
                                thread.failcount = 0    # reset fail count to zero if this move was accepted
                        elif thread.commit1 != thread.commit2 and thread.commit1 != 'fail' and thread.commit2 != 'fail':  # valid transition path, update 'last_valid' attribute
                            if settings.cleanup:
                                if os.path.exists(thread.basename + '_' + thread.last_valid + '_fwd.nc'):
                                    os.remove(thread.basename + '_' + thread.last_valid + '_fwd.nc')
                                if os.path.exists(thread.basename + '_' + thread.last_valid + '_bwd.nc'):
                                    os.remove(thread.basename + '_' + thread.last_valid + '_bwd.nc')
                            thread.last_valid = thread.suffix
                            thread.accept_moves += 1
                            thread.failcount = 0        # reset fail count to zero if this move was accepted
                        elif not settings.eps_settings and settings.cleanup:  # shooting move did not pass, so we can immediately delete the trajectories
                            if os.path.exists(thread.basename + '_' + thread.suffix + '_fwd.nc'):
                                os.remove(thread.basename + '_' + thread.suffix + '_fwd.nc')
                            if os.path.exists(thread.basename + '_' + thread.suffix + '_bwd.nc'):
                                os.remove(thread.basename + '_' + thread.suffix + '_bwd.nc')
                        cleanthread(thread, settings)
                local_index += 1

            pickle.dump(settings.allthreads, open('restart.pkl', 'wb'), protocol=2)  # dump information required to restart ATESA
            if not settings.itinerary and not settings.DEBUGMODE:
                time.sleep(60)                                  # delay 60 seconds before checking for job status again

            if not settings.itinerary and not settings.running and not settings.eps_settings and not settings.find_ts:
                acceptances = [(100 * thread.accept_moves / thread.total_moves) for thread in settings.allthreads]
                open(settings.logfile, 'a').write('\nItinerary and running lists are empty.\nAimless shooting is complete! '
                                          'The highest acceptance ratio for any thread was ' + str(max(acceptances)) +
                                          '%.\nSee as.out in the working directory for results.')
                open(settings.logfile, 'a').close()
                break
            elif not settings.itinerary and not settings.running and not settings.find_ts and settings.eps_settings:
                open(settings.logfile, 'a').write('\nItinerary and running lists are empty.\nEquilibrium path sampling is complete!'
                                          '\nSee eps_results.out in the working directory for results.')
                open(settings.logfile, 'a').close()
                break
            elif not settings.itinerary and not settings.running and settings.find_ts and not settings.eps_settings:
                acceptances = [(100 * thread.accept_moves / thread.total_moves) for thread in settings.allthreads]
                if any([(acceptance > 0) for acceptance in acceptances]):
                    open(settings.logfile, 'a').write('\nItinerary and running lists are empty.\nfind_ts is complete! '
                                                      'The highest acceptance ratio for any thread was ' +
                                                      str(max(acceptances)) + '%.\nSee status.txt in the working '
                                                      'directory for results.')
                    open(settings.logfile, 'a').close()
                    break
                else:
                    find_ts_loop(settings)


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


def cv_analysis_func(settings):
    """
    Produce a subdirectory inside the working directory containing a file for each thread, which in turn contain columns
    for every CV, giving the value of that CV vs. thread step.

    Obtains information on thread history by reading through the log file.
    Note that this will produce redundant points between different files wherever threads fork from a single starting
    point. Even if there is no forking in the log file, if there is degeneracy initial points will be shared.

    Parameters
    ----------
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    None
    """
    # First make the directory
    dirName = 'cv_analysis'
    if os.path.exists(dirName):
        shutil.rmtree(dirName)  # delete old working directory
        os.makedirs(dirName)    # make a new one
    else:
        os.makedirs(dirName)

    pattern1 = re.compile('.*_[0-9]+_1$')   # pattern matching strings ending in _X_1 where X is any integer
    pattern2 = re.compile('.*_1$')          # pattern matching strings ending in _1
    pattern3 = re.compile('.*_[0-9]+$')      # pattern matching strings ending in _X where X is any integer

    update_progress(0, message='Tabulating thread histories')
    lines = len(open(settings.logfile,'r').readlines())
    count = 0
    keeptrack = [[]]  # initialize nested list for keeping track of contributors to each file
    # Next, comb through the log file and dynamically write files as necessary.
    for logline in open(settings.logfile,'r').readlines():
        # Strategy here is to look for lines starting with "Writing forward batch file for ", followed by the name.
        # Choose which file to append to by working backwards through the name string.
        if 'Writing forward batch file for ' in logline:
            thisname = logline.replace('Writing forward batch file for ','')    # get thread name including suffix
            thisname = thisname.replace('\n','')
            if pattern1.match(thisname):        # move name ends in _X_1 where X is any integer
                # Strip _X_1 from thisname
                temp = thisname[:-2]
                flag = False
                while not flag:
                    if temp[-1] == '_':
                        flag = True
                    temp = temp[:-1]
                keeptrack_index = 0
                flag = False
                for thread in keeptrack:    # if already found a move obtainable by omitting the _X_1, forked from that
                    if thread.__contains__(temp):
                        keeptrack[keeptrack_index].append(thisname)
                        flag = True
                    keeptrack_index += 1
                if not flag:                # else this is a top-level move
                    keeptrack.append([thisname])  # make new entry in keeptrack for this top-level move
            elif pattern2.match(thisname):  # move name ends in _1
                keeptrack.append([thisname])      # make new entry in keeptrack for this top-level move
            elif pattern3.match(thisname):  # ends in _X where X is any integer (non-1, since 1 is caught above)
                # Subtract one from terminal integer in thisname
                temp = thisname
                terminal_int = ''
                flag = False
                while not flag:
                    if temp[-1] == '_':
                        flag = True
                    terminal_int = temp[-1] + terminal_int
                    temp = temp[:-1]
                terminal_int = str(int(terminal_int[1:]) - 1)   # do the actual subtraction
                temp = temp + '_' + terminal_int
                keeptrack_index = 0
                flag = False
                for thread in keeptrack:    # if already found a move obtainable by subtracting one, continued from that
                    if thread.__contains__(temp):
                        keeptrack[keeptrack_index].append(thisname)
                        flag = True
                    keeptrack_index += 1
                if not flag:                # else something went wrong
                    sys.exit('Error: shooting move in ' + settings.logfile + ': ' + thisname + ' doesn\'t fit any thread. Is the log file'
                         ' corrupted or incomplete? If you\'re sure it\'s not, please raise a GitHub issue!')
            else:
                sys.exit('Error: shooting move in ' + settings.logfile + ': ' + thisname + ' is formatted incorrectly. Is the log file '
                         'corrupted or incomplete? If you\'re sure it\'s not, please raise a GitHub issue!')
        count += 1
        update_progress(count / lines, message='Tabulating thread histories')

    update_progress(0, message='Writing CV output files')
    lines = len(keeptrack)
    count = 0
    for thread in keeptrack:
        basename = os.path.commonprefix(thread)
        with open(dirName + '/' + basename + 'cvs.out','w') as f:
            # Obtain each CV for each file in thread, report them row-by-row for each file
            for filename in thread:
                filename += '_init_fwd.rst'
                cvs = candidatevalues(filename, settings=settings)
                f.write(cvs)
                f.write('\n')
            f.close()
        count += 1
        update_progress(count / lines, message='Writing CV output files')


def zip_for_transfer_func(settings):
    """
    Identify the files necessary to continue aimless shooting and/or analysis from the current state and zip them up for
    transfer to another directory or filesystem.

    The goal of this function is to make it easy to transfer the minimum amount of data from one directory or HPC
    cluster to another without losing any functionality with regard to further aimless shooting path sampling or future
    analysis scripts like committor analysis or equilibrium path sampling. It uses the zipfile and zlib libraries.

    The files to be transfered include: shooting point coordinate files; trajectory files for last-accepted moves for
    each Thread; the restart pickle file itself; the parameter/topology file; and the log and output files. Files from
    previous analysis runs, like rc_eval.out files or committor analysis directories, are excluded.

    Parameters
    ----------
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    None
    """
    import zipfile
    import zlib     # necessary to use ZIP_DEFLATED (gzip-compatible) compression

    if os.path.exists('transfer.zip'):  # delete existing archive it is exists
        print('Deleting existing transfer.zip')
        os.remove('transfer.zip')

    print('Zipping necessary files from working directory: ' + settings.working_directory)
    update_progress(0, message='Zipping')

    transfer_zip = zipfile.ZipFile('transfer.zip', 'w')             # define empty zip file
    num_files = len(os.listdir(settings.working_directory))         # for progress bar
    count = 0                                                       # for progress bar

    # Flags indicating successful zipping of each component
    flag_pkl = False
    flag_log = False
    flag_out = False
    flag_top = False

    for file in os.listdir(settings.working_directory):             # walk through contents of working_directory
        if file.endswith('_init_fwd.rst'):                          # shooting move coordinate files
            transfer_zip.write(file, compress_type = zipfile.ZIP_DEFLATED)  # add to archive with zlib compression
        elif (file in [thread.basename + '_' + thread.last_valid + '_fwd.nc' for thread in settings.allthreads]) or\
             (file in [thread.basename + '_' + thread.last_valid + '_bwd.nc' for thread in settings.allthreads]):
            transfer_zip.write(file, compress_type = zipfile.ZIP_DEFLATED)  # add to archive with zlib compression
        elif file == 'restart.pkl' or file == settings.logfile or file == 'as.out' or file == settings.topology:
            transfer_zip.write(file, compress_type = zipfile.ZIP_DEFLATED)  # add to archive with zlib compression
            if file == 'restart.pkl':
                flag_pkl = True
            elif file == settings.logfile:
                flag_log = True
            elif file == 'as.out':
                flag_out = True
            elif file == settings.topology:
                flag_top = True
        count += 1
        update_progress(count / num_files, message='Zipping')

    # todo: technically could check if these things are present first, save some time, especially when a lot of data is at hand.
    if not flag_pkl:
        sys.exit('Error: could not zip restart.pkl. Check that it is present and accessible in the working directory.')
    elif not flag_log:
        sys.exit('Error: could not zip ' + settings.logfile + '. Check that it is present and accessible in the working directory.')
    elif not flag_out:
        sys.exit('Error: could not zip as.out. Check that it is present and accessible in the working directory.')
    elif not flag_top:
        sys.exit('Error: could not zip topology file ' + settings.topology + '. Check that it is present and accessible in the working directory.')

    # Don't forget to add completed halves of shooting moves that are still running, should any exist
    print('Adding partially-completed shooting move data...')
    update_progress(0, message='Zipping')
    num_threads = len(settings.allthreads)
    count = 0
    for thread in settings.allthreads:
        if (thread.commit1 and not thread.commit2):
            transfer_zip.write(thread.basename + '_' + thread.suffix + '_fwd.nc', compress_type=zipfile.ZIP_DEFLATED)
        elif (thread.commit2 and not thread.commit1):
            transfer_zip.write(thread.basename + '_' + thread.suffix + '_bwd.nc', compress_type=zipfile.ZIP_DEFLATED)
        count += 1
    update_progress(count / num_threads, message='Zipping')


def initialize_eps(settings):
    """
    Initialize global settings for EPS runs. This is only called once in the code (in the if __name__ == '__main__'
    section) but is pulled into a separate function here for the sake of cleanliness and unit testing.

    Parameters
    ----------
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    None
    """
    settings.n_windows = settings.eps_settings[0]
    settings.k_beads = settings.eps_settings[1]
    settings.rc_min = settings.eps_settings[2]
    settings.rc_max = settings.eps_settings[3]
    settings.traj_length = settings.eps_settings[4]
    settings.overlap = settings.eps_settings[5]  # used when setting eps_min/max values, not in defining eps_windows
    if settings.traj_length % settings.k_beads:
        sys.exit('Error: in eps_settings, traj_length is not divisible by k_beads')
    eps_in_template = 'eps_in.tpl'
    for k in range(settings.k_beads - 1):
        template = settings.env.get_template(eps_in_template)
        filled = template.render(nstlim=str(int((k + 1) * settings.traj_length / settings.k_beads)),
                                 ntwx=str(int(settings.traj_length / settings.k_beads)))
        with open(settings.working_directory + '/eps' + str(k + 1) + '.in', 'w') as newfile:
            newfile.write(filled)
            newfile.close()
    window_width = (settings.rc_max - settings.rc_min) / settings.n_windows
    settings.eps_windows = [settings.rc_min + window_width * n for n in range(settings.n_windows)] + [
        settings.rc_min + window_width * (settings.n_windows)]
    if not settings.restart:  # since if restart, this file should already exist; overwrites for resample
        open('eps_results.out', 'w').write('Lower boundary of RC window; Upper boundary; RC value\n')
        open('eps_results.out', 'a').close()
    elif not os.path.exists('eps_results.out'):
        sys.exit('Error: restart = True and this is an EPS run, but eps_results.out was not found in the working '
                 'directory: ' + settings.working_directory)


def handle_bootstrap(settings):
    """
    Handle tasks associated with bootstrapping convergence of the reaction coordinate produced by LMAX. This function
    has branching behavior depending on what's going on when it's called:

    If the variable settings.bootstrap_flag == True, it will just return True without running any jobs. This is so that
    once bootstrapping has returned True once, more tests will not be run even when handle_bootstrap() is called again.

    Else, if no bootstraping jobs are running or queued, it will call makebatch() and subbatch() to build and submit
    batch jobs that will run atesa_lmax.py on the current as.out file and on settings.bootstrap_n bootstrapped versions
    of that file that this function will build

    Else, if those jobs are running or queued, it will return False.

    Else, if those jobs have finished since the last time it was called, it will compare the results using the
    subfunction compare_rcs() and return a flag identifying whether the RCs are converged or not.

    Parameters
    ----------
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    bool
        True if the bootstrapped RCs agree with the full-data RC; False otherwise.
    """

    def compare_rcs(rcs):
        # Compares the RCs passed to it as strings in the list object rcs, where each object in rcs is a string that
        # can be interpreted as an RC. Returns True if the RCs all contain the same CVs and each of their coefficients
        # is within settings.bootstrap_threshold of the all-data value; False otherwise.
        # Each RC should be formatted as: coeff0 + coeff1*CV1 +/- coeff2*CV2 +/- ..., where coeffN is a positive float.
        # The first RC in rcs should be the all-data RC.
        # First, split each RC object into a sublist of terms
        splitrcs = []
        for rc in rcs:
            temp = rc.replace('- ', '+ -')  # force all operations to be addition while preserving negatives
            splitrcs.append(temp.split(' + '))
        # Iterate through RCs in splitrcs and check agreement criteria
        alldata_values = [[], []]
        bare_term_flag = False  # for bookkeeping of coefficients without CVs
        for term in splitrcs[0]:
            alldata_values[0].append(term.split('*')[0])
            try:
                alldata_values[1].append(term.split('*')[1])
            except IndexError:
                if bare_term_flag == False:
                    bare_term_flag = True
                else:
                    sys.exit('Error: handle_bootstrap.compare_rcs() was passed an RC with more than a single term '
                             'lacking a corresponding CV. This is probably developer error, not user error. As such, '
                             'please raise an issue on GitHub (https://github.com/team-mayes/atesa) including your '
                             'input file and this error message. The offending RC is: ' + splitrcs[0])
        for rc in splitrcs:
            bare_term_flag = False  # for bookkeeping of coefficients without CVs
            term_index = -1
            for term in rc:
                term_index += 1
                try:
                    if term.split('*')[1] not in alldata_values[1]:
                        print(term.split('*')[1])
                        return False  # CV absent from reference
                except IndexError:
                    if bare_term_flag == False:
                        bare_term_flag = True
                    else:
                        sys.exit('Error: handle_bootstrap.compare_rcs() was passed an RC with more than a single term '
                                 'lacking a corresponding CV. This is probably developer error, not user error. As such'
                                 ', please raise an issue on GitHub (https://github.com/team-mayes/atesa) including '
                                 'your input file and this error message. The offending RC is: ' + rc)
                if not (float(alldata_values[0][term_index]) - settings.bootstrap_threshold <= float(
                        term.split('*')[0]) <= float(alldata_values[0][term_index]) + settings.bootstrap_threshold):
                    return False  # coefficient outside of reference range
        return True  # Only happens if every other check above was passed

    # Check that a previous call to handle_bootstrap hasn't already returned True
    if settings.bootstrap_flag == True:
        return True

    # Check that bootstrapping jobs aren't currently running or queued
    queue = interact('queue', settings)
    for jobid in settings.bootstrap_jobids:
        if str(jobid) in queue:
            return False

    # Check bookkeeping variable settings.bootstrap_bookkeep to determine whether the last-submitted bootstrapping jobs
    # (if they exist) have been checked for agreement yet.
    if not settings.bootstrap_bookkeep:
        # Obtain results and check for agreement
        settings.bootstrap_bookkeep = True
        rcs = []
        for i in range(settings.bootstrap_n + 1):
            this_lines = open('bootstrap_' + str(settings.len_data) + '_' + str(i) + '.out', 'r').readlines()
            this_rc = this_lines[-1]                        # the RC is given on the last line of the output file
            this_rc = this_rc.replace('Final RC: ', '')     # to remove the bit that comes before the actual RC string
            rcs.append(this_rc)
        return compare_rcs(rcs)
    else:
        # Create and submit new bootstrap input files/batch files
        settings.bootstrap_bookkeep = False
        asoutlines = open('as.out','r').readlines()
        settings.len_data = len(asoutlines)
        temp_thread = Thread()      # make a temporary thread object to pass to makebatch
        temp_thread.type = 'bootstrap'
        makebatch(temp_thread, settings)
        # This makes files named 'bootstrap_' + str(settings.len_data) + '_' + str(i) + '.' + settings.batch_system for
        # i in range [0, settings.bootstrap_n], each of which takes as its input file the same thing but with file
        # extension .in instead. Before subbatch, I have to make those files!
        shutil.copyfile('as.out', 'bootstrap_' + str(settings.len_data) + '_0.in')
        for file_index in range(settings.bootstrap_n):
            with open('bootstrap_' + str(settings.len_data) + '_' + str(file_index + 1) + '.in', 'a') as f:
                for null in range(settings.len_data):
                    line_index = random.randint(0,settings.len_data-1)
                    f.write(asoutlines[line_index])
                f.close()
        settings.bootstrap_jobids = subbatch(Thread(), 'bootstrap', settings=settings)
        return False


def init_find_ts(settings,start_name):
    """
    Initialize and perform jobs for constructing transition state guesses from the provided input coordinate files. The
    resulting guesses should be passed back to if __name__ == '__main__' and interpreted as if they were user-supplied
    transition state structures.

    This function produces unique transition state guesses dynamically by comparing the settings.commit_fwd and
    settings.commit_bwd entries against the corresponding values in the supplied structure (only one is permitted),
    writing new simulation input files with bond-length constraints to gently force the system to cross from the basin
    it began in to the other one, running and monitoring those simulations, and then harvesting key frames from within
    those trajectories where the relevant bond lengths are at intermediate values. The value of int(settings.find_ts)
    determines the number of unique forced trajectories that are performed and harvested from.

    The produced coordinates are by no means guaranteed to be good transition state guesses, and indeed most will not
    be. Once the guesses are produced, they are automatically used to seed aimless shooting runs. Good transition state
    guesses will be those that have good (maybe 30-50%) acceptance ratios during this step. The user should submit
    find_ts ATESA runs with the expectation of obtaining transition state guesses, and then perform separate ATESA runs
    using the good guesses obtained by find_ts as input coordinates.

    Parameters
    ----------
    settings : Namespace
        Global settings Namespace object.
    start_name : list
        A one-length list containing a string indicating the name of the initial coordinate file to develop the
        transition state guess from.

    Returns
    -------
    start_name_out : list
        A list containing strings corresponding to the names of the newly produced transition state guesses.
    """
    # First step is to identify the basin that the provided structure is in and throw an error if it's in neither.
    open(settings.logfile, 'a').write('\nInitializing find_ts transition state search.')
    commit = standalone_checkcommit(start_name[0], settings)
    if commit == 'fail':
        sys.exit('Error: the coordinates provided during a find_ts run must represent a structure in either the fwd '
                 'or bwd basin, but the coordinate file: ' + start_name[0] + ' is in neither. If it is a transition '
                 'state guess, you should set "find_ts = 0" (default)')
    elif commit == 'fwd':
        other_basin_define = settings.commit_define_bwd
        dir_to_check = 'bwd'
    elif commit == 'bwd':
        other_basin_define = settings.commit_define_fwd
        dir_to_check = 'fwd'
    else:
        sys.exit('Error: internal error in checkcommit(); did not return valid output.')
    # Then write the restraint file
    open(settings.logfile, 'a').write('\nWriting restraint file find_ts_restraints.DISANG')
    open('find_ts_restraints.disang', 'w').write('')  # initialize file
    with open('find_ts_restraints.disang', 'a') as f:
        f.write('DISANG restraint file produced by ATESA with find_ts > 0 to produce transition state guesses\n')
        for def_index in range(len(other_basin_define[0])):
            extra = 0   # additional distance to add to basin definition to push *into* basin rather than to its edge
            if other_basin_define[3][def_index] == 'lt':
                extra = -0.1
            elif other_basin_define[3][def_index] == 'gt':
                extra = 0.1
            f.write(' &rst\n')
            f.write('  iat=' + other_basin_define[0][def_index][1:] + ',' + other_basin_define[1][def_index][1:] + ',\n')
            f.write('  r1=0, r2=' + str(other_basin_define[2][def_index] + extra) + ', r3=' + str(other_basin_define[2][def_index] + extra) + ', r4=' + str(other_basin_define[2][def_index] + extra + 2) + ',\n')
            f.write('  rk2=500, rk3=500,\n')
            f.write(' &end\n')
        f.close()
    # Check that the input file for these simulations exists
    if not os.path.exists(settings.home_folder + '/' + 'input_files/find_ts.in'):
        sys.exit('Error: could not locate input file \'find_ts.in\' in ' + settings.home_folder + '/' + 'input_files')
    # Now write and submit the batch file(s)...
    settings.running = []       # a list of currently running threads
    settings.allthreads = []    # a list of all threads regardless of status
    batch = 'batch_' + settings.batch_system + '.tpl'
    for ts_job in range(settings.find_ts):
        name = str(ts_job + 1)
        template = settings.env.get_template(batch)
        filled = template.render(name=name + '_find_ts', nodes=settings.prod_nodes, taskspernode=settings.prod_ppn,
                                 walltime=settings.prod_walltime,
                                 solver='sander', inp=settings.home_folder + '/input_files/find_ts.in',
                                 out=name + '_find_ts.out', prmtop=settings.topology,
                                 inpcrd=start_name[0], rst=name + '_find_ts.rst', nc=name + '_find_ts.nc',
                                 mem=settings.prod_mem, working_directory=settings.working_directory)
        open(settings.logfile, 'a').write('\nWriting barrier crossing job batch file: ' + name + '_find_ts.' + settings.batch_system)
        with open(name + '_find_ts.' + settings.batch_system, 'w') as newfile:
            newfile.write(filled)
            newfile.close()
        thread = spawnthread(name, 'find_ts', settings=settings)
        thread.jobid1 = subbatch(thread, settings=settings)
        settings.running.append(thread)
    # Every minute, check if all of the jobs have committed to the appropriate basin or otherwise terminated
    if settings.batch_system == 'pbs':
        cancel_command = 'qdel'
    elif settings.batch_system == 'slurm':
        cancel_command = 'scancel'
    else:
        sys.exit('Error: invalid batch_system type: ' + str(settings.batch_system))
    while settings.running:
        time.sleep(60)
        output = interact('queue', settings)
        for thread in settings.running:
            thread.commit1 = checkcommit(thread.basename, direction='find_ts', settings=settings)
            if thread.jobid1 in output:
                if thread.commit1 == dir_to_check:
                    open(settings.logfile, 'a').write('\nBarrier crossing job: ' + thread.basename + '_find_ts.' +
                                                      settings.batch_system + ' has committed to the opposite basin and is '
                                                                              'being terminated.')
                    process = subprocess.Popen([cancel_command, thread.jobid1], stdout=subprocess.PIPE)
                    (subprocess_output, err) = process.communicate()  # doesn't do anything, I think
                elif thread.commit1:
                    thread.commit1 = ''
            else:
                if not thread.commit1 == dir_to_check:
                    thread.commit1 = 'fail'
                    open(settings.logfile, 'a').write('\nBarrier crossing job: ' + thread.basename + '_find_ts.' +
                                                       settings.batch_system + ' has failed to achieve crossing.')
        local_index = 0                                 # robustly remove completed jobs from settings.running
        done_flag = False
        while not done_flag:
            thread = settings.running[local_index]
            if thread.commit1:
                settings.running.remove(thread)
                local_index -= 1
            local_index += 1
            if local_index == len(settings.running):
                done_flag = True

    # After all the jobs are finished, discard failures with a warning and harvest candidate TS's from the remainders
    if not dir_to_check in [thread.commit1 for thread in settings.allthreads]:
        sys.exit('Error: Of the (' + str(settings.find_ts) + ') attempt(s) at finding a transition state, none of the '
                 'initial barrier crossing trajectories reached the opposite basin. The most likely explanation is '
                 'that the definition of the target basin (' + dir_to_check + ') is unsuitable for some reason. Please '
                 'check the produced trajector(y/ies) and output file(s) in the working directory ('
                 + settings.working_directory + ') and either modify the basin definitions or the find_ts input file ('
                 + settings.home_folder + '/input_files/find_ts.in) accordingly.')
    else:
        for thread in settings.allthreads:
            if not thread.commit1 == dir_to_check:
                open(settings.logfile, 'a').write('\nWarning: find_ts thread ' + thread.basename + ' did not commit to '
                                                  'the opposite basin and is being discarded.')
    # Harvest TSs by inspecting values of basin-defining bond lengths vs. frame number
    # Since there's no fundamental requirement that the defining bond lengths in the fwd and bwd basin definitions be
    # the same, I'll look only at the lengths that were restrained during the find_ts barrier crossing step.

    # Quick helper function definition
    def my_integral(params, my_list):
        # Evaluate a numerical integral of the data in list my_list on the range (params[0],params[1]), where the
        # values in params should be integers referring to indices of my_list.
        partial_list = my_list[int(params[0]):int(params[1])]
        return numpy.trapz(partial_list)

    # And aother quick function to find the bounds of my_integral that maximize its output
    def my_bounds_opt(objective, data):
        list_of_bounds = []
        for left_bound in range(len(data)-1):
            for right_bound in range(left_bound+1,len(data)+1):
                if right_bound - left_bound > 1:
                    list_of_bounds.append([left_bound,right_bound])
        output = argparse.Namespace()
        output.best_max = objective(list_of_bounds[0], data)
        output.best_bounds = list_of_bounds[0]
        for bounds in list_of_bounds:
            this_result = objective(bounds, data)
            if this_result > output.best_max:
                output.best_max = this_result
                output.best_bounds = bounds
        return output

    output = []
    for thread in settings.allthreads:
        traj = pytraj.iterload(thread.basename + '_find_ts.nc', settings.topology)
        this_lengths = []
        for def_index in range(len(other_basin_define[0])):
            this_lengths.append(pytraj.distance(traj, mask=other_basin_define[0][def_index] + ' ' + other_basin_define[1][def_index]))  # appends a list of distances for this traj to this_lengths
        # Now look for the TS by identifying the region in the trajectory with all of the bond lengths at intermediate
        # values (say, 0.25 < X < 0.75 on a scale of 0 to 1), preferably for several frames in a row.
        norm_lengths = []
        scored_lengths = []
        for lengths in this_lengths:
            normd = [(this_len - min(lengths))/(max(lengths) - min(lengths)) for this_len in lengths] # normalize lengths to between 0 and 1
            norm_lengths.append(normd)
            scored_lengths.append([(-(this_len - 0.5)**2 + 0.0625) for this_len in normd])            # scored normalized lengths on parabola
        sum_of_scores = []
        for frame_index in range(len(scored_lengths[0])):       # sum together scores from each distance at each frame
            sum_of_scores.append(sum([[x[i] for x in scored_lengths] for i in range(len(scored_lengths[0]))][frame_index]))
        # Now I want to find the boundaries between which the TS is most likely to reside. To do this I'll perform a 2D
        # optimization on the integral to find the continuous region that is most positively scored
        opt_result = my_bounds_opt(my_integral, sum_of_scores)
        # If all of sum_of_scores is negative we still want to find a workable max. Also, if there are fewer than five
        # candidate frames we want to test more than that. Either way, shift scores up by 0.1 and try again:
        while opt_result.best_max <= 0 or int(opt_result.best_bounds[1] - opt_result.best_bounds[0]) < 5:
            sum_of_scores = [(item + 0.1) for item in sum_of_scores]
            opt_result = my_bounds_opt(my_integral, sum_of_scores)
        # opt_result.best_bounds now contains the frame indices bounding the region of interest. I want to use pytraj to
        # extract each of these frames as a .rst7 coordinate file and return their names. I'll put a cap on the number
        # of frames that this is done for at, say, 50, so as to avoid somehow ending up with a huge number of candidate
        # TS structures to test.
        open(settings.logfile, 'a').write('\nfind_ts thread ' + thread.basename + ' contains ' +
                                          str(int(opt_result.best_bounds[1] - opt_result.best_bounds[0]) + 1) +
                                          ' candidate transition state frames')
        if int(opt_result.best_bounds[1] - opt_result.best_bounds[0] + 1) > 50:
            open(settings.logfile, 'a').write(' of which 50 with the closest possible approximation of even spacing '
                                              'will be extracted for testing.')
            frame_indices = [int(ii) for ii in numpy.linspace(opt_result.best_bounds[0], opt_result.best_bounds[1], 50)]
        else:
            open(settings.logfile, 'a').write(', all of which will be extracted for testing.')
            frame_indices = [int(ii) for ii in range(opt_result.best_bounds[0], opt_result.best_bounds[1]+1)]
        for frame_index in frame_indices:
            pytraj.write_traj(thread.basename + '_' + thread.type + '_' + str(frame_index) + '.rst7', traj, frame_indices=[frame_index], options='multi',overwrite=True)
            try:

                os.rename(thread.basename + '_' + thread.type + '_' + str(frame_index) + '.rst7.1',
                          thread.basename + '_' + thread.type + '_' + str(frame_index) + '.rst7')
            except FileNotFoundError:
                if not os.path.exists(thread.basename + '_' + thread.type + '_' + str(frame_index) + '.rst7'):
                    sys.exit('Error: attempted to write file ' + thread.basename + '_' + thread.type + '_' +
                             str(frame_index) + '.rst7, but was unable to. Please ensure that you have adequate '
                             'permissions to write to the working directory: ' + settings.working_directory)
            output.append(thread.basename + '_' + thread.type + '_' + str(frame_index) + '.rst7')
        open('debug_' + thread.basename, 'w').write('opt_result.best_bounds: ' + str(opt_result.best_bounds) +
                                                    '\nopt_result.best_max: ' + str(opt_result.best_max) +
                                                    '\nsum_of_scores: ' + str(sum_of_scores))


    settings.running = []       # clean up running and allthreads before handing off back to main code
    settings.allthreads = []

    open(settings.logfile, 'a').write('\nfind_ts transition state search complete! Passing ' + str(len(output)) +
                                      ' candidate transition state structures forward for testing.')

    return output


def find_ts_loop(settings):
    """
    Interpret results of a failed find_ts loop (which consists of submitting TS guesses to main_loop, having all threads
    terminate, and having no acceptances in any thread) and respond accordingly. This function will either write new
    TS guesses in an attempt to obtain a working TS, or communicate helpfully with the user, depending on the results.

    Parameters
    ----------
    settings : Namespace
        Global settings Namespace object.

    Returns
    -------
    None
    """
    output = []
    for basename in [(str(ii + 1) + '_find_ts') for ii in range(int(settings.find_ts))]:
        results = []
        indices = []
        for thread in [this_thread for this_thread in settings.allthreads if basename in this_thread.basename]:
            for history_str in thread.history:
                results.append(history_str[-1])     # get just the result code from each move in this thread
            indices.append(int(thread.basename.replace(basename + '_', '').replace('.rst7', '')))   # index from original init_find_ts trajectory
        traj = pytraj.iterload(basename + '.nc', settings.topology)
        if 'B' in results and 'F' not in results:   # threads went backward but never forward
            new_indices = [(this_index + max(indices) - min(indices) + 1) for this_index in indices if
                           (this_index + max(indices) - min(indices) + 1) < traj.n_frames]
            if new_indices:
                for frame_index in new_indices:
                    pytraj.write_traj(basename + '_' + str(frame_index) + '.rst7', traj,
                                      frame_indices=[frame_index], options='multi', overwrite=True)
                    try:

                        os.rename(basename + '_' + str(frame_index) + '.rst7.1',
                                  basename + '_' + str(frame_index) + '.rst7')
                    except FileNotFoundError:
                        if not os.path.exists(basename + '_' + str(frame_index) + '.rst7'):
                            sys.exit('Error: attempted to write file ' + basename + '_' + str(frame_index) +
                                     '.rst7, but was unable to. Please ensure that you have adequate permissions to '
                                     'write to the working directory: ' + settings.working_directory)
                    output.append(basename + '_' + str(frame_index) + '.rst7')
            else:
                open(settings.logfile,'a').write('\nfind_ts thread ' + basename + ' failed to produce a transition '
                                                 'state for reason: unable to obtain commitment to forward basin during'
                                                  ' aimless shooting. This is a strange error and may indicate highly '
                                                  'non-convergent simulation settings. If this error occurs multiple '
                                                  'times, the user is encouraged to manually inspect the simulations '
                                                  'produced for this thread for errors or unusual behavior.')
        elif 'F' in results and 'B' not in results:  # threads went forward but never backward
            new_indices = [(this_index - max(indices) + min(indices) - 1) for this_index in indices if
                           (this_index - max(indices) + min(indices) + 1) >= 0]
            if new_indices:
                for frame_index in new_indices:
                    pytraj.write_traj(basename + '_' + str(frame_index) + '.rst7', traj,
                                      frame_indices=[frame_index], options='multi', overwrite=True)
                    try:

                        os.rename(basename + '_' + str(frame_index) + '.rst7.1',
                                  basename + '_' + str(frame_index) + '.rst7')
                    except FileNotFoundError:
                        if not os.path.exists(basename + '_' + str(frame_index) + '.rst7'):
                            sys.exit('Error: attempted to write file ' + basename + '_' + str(frame_index) + '.rst7'
                                     ', but was unable to. Please ensure that you have adequate permissions to '
                                     'write to the working directory: ' + settings.working_directory)
                    output.append(basename + '_' + str(frame_index) + '.rst7')
            else:
                open(settings.logfile,'a').write('\nfind_ts thread ' + basename + ' failed to produce a transition '
                                                 'state for reason: unable to obtain commitment to backward basin '
                                                 'during aimless shooting. This is a strange error and may indicate '
                                                 'highly non-convergent simulation settings. If this error occurs '
                                                 'multiple times, the user is encouraged to manually inspect the '
                                                 'simulations produced for this thread for errors or unusual behavior.')
        elif 'F' in results and 'B' in results:     # some threads went only forward and some went only backwards
            nested_history = []
            for thread in [this_thread for this_thread in settings.allthreads if basename in this_thread.basename]:
                temp_results = []
                for history_str in thread.history:
                    temp_results.append(history_str[-1])
                nested_history.append(temp_results)
            for index_index in range(len(indices) - 1):
                if (('F' not in nested_history[index_index] and 'B' not in nested_history[index_index + 1]) or
                   ('B' not in nested_history[index_index] and 'F' not in nested_history[index_index + 1])) and\
                   indices[index_index] + 1 < indices[index_index + 1]:
                    for frame_index in range(indices[index_index] + 1, indices[index_index + 1]):
                        pytraj.write_traj(basename + '_' + str(frame_index) + '.rst7', traj,
                                          frame_indices=[frame_index], options='multi', overwrite=True)
                        try:

                            os.rename(basename + '_' + str(frame_index) + '.rst7.1',
                                      basename + '_' + str(frame_index) + '.rst7')
                        except FileNotFoundError:
                            if not os.path.exists(basename + '_' + str(frame_index) + '.rst7'):
                                sys.exit('Error: attempted to write file ' + basename + '_' + str(frame_index) + '.rst7'
                                         ', but was unable to. Please ensure that you have adequate permissions to '
                                         'write to the working directory: ' + settings.working_directory)
                        output.append(basename + '_' + str(frame_index) + '.rst7')
                elif 'F' in nested_history[index_index] and 'B' in nested_history[index_index]:
                    open(settings.logfile, 'a').write('\nStructure ' + basename + '_' + str(indices[index_index])
                                                      + '.rst7 appears to be a suitable TS, but may have a very low '
                                                      'acceptance ratio.')
                elif (('F' not in nested_history[index_index] and 'B' not in nested_history[index_index + 1]) or
                     ('B' not in nested_history[index_index] and 'F' not in nested_history[index_index + 1])) and\
                     not (indices[index_index] + 1 < indices[index_index + 1]):
                    open(settings.logfile, 'a').write('\nTrajectory ' + basename + ' appears to have a transition state'
                                                      ' between frames ' + str(indices[index_index]) + ' and ' +
                                                      str(indices[index_index + 1]) + '. You may be able to improve the'
                                                      ' ability of find_ts to obtain this transition state by reducing '
                                                      'the simulation step size or steps between writes to the output '
                                                      'trajectory in find_ts.in.')

    for structure in output:
        thread = spawnthread(structure, suffix='1', settings=settings)  # spawn a new thread with default settings
        thread.prmtop = settings.topology                               # set prmtop filename for the thread
        settings.itinerary.append(thread)                               # submit it to the itinerary

    if output:
        open(settings.logfile, 'a').write('\nfind_ts transition loop complete. Passing ' + str(len(output)) +
                                          ' more candidate transition state structures forward for testing.')
        main_loop(settings)     # pass back to main_loop having added new jobs to itinerary
    else:
        open(settings.logfile, 'a').write('\nUnable to produce a structure with non-zero acceptance ratio from the '
                                          'supplied initial guess and given settings. Perhaps a different initial '
                                          'structure, more relaxed termination criteria, or other change would help. '
                                          'Exiting.')
        sys.exit()              # not an error, per se.



def main():
    """
    Initialize settings object and call the appropriate functions in accordance with user-defined options.

    This function is only called in the event that atesa.py is run directly.

    Parameters
    ----------

    Returns
    -------
    None
    """
    global settings

    # Get directory from which the code was called
    called_path = os.getcwd()

    # Parse arguments from command line using argparse
    parser = argparse.ArgumentParser(
        description='Perform aimless shooting according to the settings given in the input file.')
    parser.add_argument('-O', action='store_true',
                        help='flag indicating that existing working_directory should be overwritten if it exists.')
    parser.add_argument('-i', metavar='input_file', type=str, nargs=1, default='as.in',
                        help='input filename; see documentation for format. Default=\'as.in\'')
    parser.add_argument('-w', metavar='working_directory', type=str, nargs=1, default=os.getcwd() + '/as_working',
                        help='working directory. Default=\'`pwd`/as_working\'')
    arguments = vars(parser.parse_args())  # Retrieves arguments as a dictionary object

    overwrite = arguments.get('O')

    path = arguments.get('i')  # parser drops dashes ('-') from variable names, so the -i argument is retrieved as such
    if type(path) == list:  # handles inconsistency in format when the default value is used vs. when a value is given
        path = path[0]
    if not os.path.exists(path):
        sys.exit('Error: cannot find input file \'' + path + '\'')

    input_file = open(path)
    input_file_lines = [i.strip('\n').split(' ') for i in input_file.readlines() if i]  # if i skips blank lines
    input_file.close()

    # Initialize default values for all the input file entries, to be overwritten by the actual contents of the file
    initial_structure = 'inpcrd'  # Initial structure filename
    if_glob = False  # True if initial_structure should be interpreted as a glob argument
    topology = 'prmtop'  # Topology filename
    n_adjust = 50  # Max number of frames by which each step deviates from the previous one
    batch_system = 'slurm'  # Batch system type, Slurm or PBS
    working_directory = arguments.get('w')  # Working directory for aimless shooting calculations
    restart_on_crash = False  # If a thread crashes during initialization, should it be resubmitted?
    max_fails = -1  # Number of unaccepted shooting moves before a thread is killed; a negative value means "no max"
    max_moves = -1  # Number of moves with any result permitted before the thread is terminated
    max_accept = -1  # Number of accepted moves permitted before the thread is terminated
    degeneracy = 1  # Number of duplicate threads to produce for each initial structure
    init_nodes = 1  # Number of nodes on which to run initialization jobs
    init_ppn = 1  # Number of processors per node on which to run initialization jobs
    init_walltime = '01:00:00'  # Wall time for initialization jobs
    init_mem = '4000mb'  # Memory per core for intialization jobs
    prod_nodes = 1  # Number of nodes on which to run production jobs
    prod_ppn = 1  # Number of processors per node on which to run production jobs
    prod_walltime = '01:00:00'  # Wall time for production jobs
    prod_mem = '4000mb'  # Memory per core for production jobs
    resample = False  # If True, don't run any new simulations; just rewrite as.out based on current settings and existing data
    fork = 1  # Number of new threads to spawn after each successful shooting move. Each one gets its own call to pickframe()
    home_folder = os.path.dirname(os.path.realpath(sys.argv[0]))  # Directory containing templates and input_files folders
    always_new = False  # Pick a new shooting move after every move, even if it isn't accepted?
    rc_definition = ''  # Defines the equation of the reaction coordinate for an rc_eval run
    rc_minmax = ''  # Aimless shooting output file from which to determine minimum and maximum CV values during rc_eval
    as_out = 'as.out'  # Output filename to identify minimum and maximum values of CVs from (replaces rc_minmax)
    candidateops = ''  # Defines the candidate order parameters to output into the as.out file
    literal_ops = False  # Determines whether to interpret candidateops as literal python code separated by semi-colons. This isn't a user-defined option, it's handled automatically based on the format of candidateops
    commit_define_fwd = []  # Defines commitment to fwd basin
    commit_define_bwd = []  # Defines commitment to bwd basin
    committor_analysis = ''  # Variables to pass into committor analysis.
    restart = False  # Whether or not to restart an old AS run located in working_directory
    groupfile = 0  # Number of jobs to submit in one groupfile, if necessary
    groupfile_max_delay = 3600  # Time in seconds to allow groupfiles to remain in construction before submitting
    include_qdot = False  # Flag to include order parameter rate of change values in output
    eps_settings = []  # Equilibrium Path Sampling: [n_windows, k_beads, rc_min, rc_max, traj_length, overlap]
    eps_dynamic_seed = ''  # Flag to seed empty windows using beads from other trajectories during EPS
    minmax_error_behavior = 'accept'  # Indicates behavior when an reduced OP value is outside range [0,1]
    zip_for_transfer = False  # Indicates that we should simply zip up necessary files and quit
    cv_analysis = False  # Indicates that we just want to produce a CV analysis output folder
    skip_log = False  # Indicates that the user knows the log file is missing or broken and can't be used for analysis scripts
    bootstrap_threshold = 0.1   # Threshold for comparison of model RC coefficients during RC bootstrapping
    bootstrap_n = 0     # Number of bootstrapped models to build during RC bootstrapping
    cvs_in_rc = 3   # Number of CVs to include in RCs produced in the course of sampling
    find_ts = 0     # Indicates that the supplied coordinates are not TS guesses but either reactant or product structures, and need to be made into guesses
    cleanup = True     # If True, trajectory files are deleted when no longer needed; not totally implemented for groupfile > 0 # todo: test
    # todo: add option for solver other than sander.MPI (edit templates to accomodate arbitrary solver name including .MPI if desired)
    # todo: (ambitious?) add support for arbitrary MD engine? I'm unsure whether I think this task is trivial or monumental

    logfile = 'as.log'  # Name of log file. Not user-set at present.
    DEBUGMODE = False   # Used only for testing, causing batch system-facing functions to return spoofed output

    if type(
            working_directory) == list:  # handles inconsistency in format when the default value is used vs. when a value is given
        working_directory = working_directory[0]

    def str2bool(var):  # Function to convert string "True" or "False" to corresponding boolean
        return str(var).lower() in ['true']  # returns False for anything other than "True" (case-insensitive)

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
                null = int(entry[2])
            except ValueError:
                sys.exit('Error: n_adjust must be an integer')
            n_adjust = int(entry[2])
        elif entry[0] == 'batch_system':
            if not entry[2].lower() == 'pbs' and not entry[2].lower() == 'slurm':
                sys.exit('Error: batch_system must be either pbs or slurm')
            batch_system = entry[2].lower()
        elif entry[0] == 'working_directory':
            if not '/' in entry[2]:  # interpreting as a subfolder of cwd
                working_directory = os.getcwd() + '/' + entry[2]
            else:  # interpreting as an absolute path
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
        elif entry[0] == 'candidate_cv' or entry[0] == 'candidate_op':
            try:
                null = isinstance(ast.literal_eval(entry[2])[0], list)
            except SyntaxError:
                literal_ops = True  # not a nested list, implies that this is an explicit definition
                full_entry = ''  # string to write all the input into
                for index in range(2, len(entry)):  # full input is contained in entry[index] for all index >= 2
                    full_entry += entry[index] + ' '  # reconstruct the original full string with spaces included
                full_entry = full_entry[:-1]  # remove trailing ' '
                candidateops = full_entry.split(';')  # each entry should be a string interpretable as an OP
            if not literal_ops:  # only do this stuff if the OPs are not given literally
                if ' ' in entry[2]:
                    sys.exit('Error: candidate_cv cannot contain whitespace (\' \') characters')
                candidateops = ast.literal_eval(entry[2])
                if len(candidateops) < 2:
                    sys.exit('Error: candidate_cv must have length >= 2')
                if not isinstance(candidateops[0], list) and not len(candidateops) == 2:
                    sys.exit('Error: if defining candidate_cv implicitly, exactly two inputs in a list are required')
                if not isinstance(candidateops[0], list) and not isinstance(candidateops[0], str):
                    sys.exit(
                        'Error: if defining candidate_cv implicitly, the first entry must be a string (including quotes)')
                if not isinstance(candidateops[0], list) and not isinstance(candidateops[1], str):
                    sys.exit(
                        'Error: if defining candidate_cv implicitly, the second entry must be a string (including quotes)')
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
            if not ([len(num) for num in entry[2].split(':')] == [2, 2, 2]):
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
            if not ([len(num) for num in entry[2].split(':')] == [2, 2, 2]):
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
            for i in range(len(entry) - 2):
                rc_definition += entry[i + 2] + ' '
            rc_definition = rc_definition[:-1]  # remove trailing space
        elif entry[0] == 'rc_minmax':
            if len(entry) > 3:
                sys.exit('Error: rc_minmax cannot contain whitespace (\' \') characters')
            rc_minmax = ast.literal_eval(entry[2])
            if rc_minmax:  # to allow rc_minmax = '', only perform these checks if it's not
                if not len(rc_minmax) == 2:
                    sys.exit('Error: rc_minmax must have two rows')
                for i in range(len(rc_minmax)):
                    if rc_minmax[0][i] and rc_minmax[1][i] and rc_minmax[0][i] >= rc_minmax[1][i]:
                        sys.exit(
                            'Error: values in the second row of rc_minmax must be larger than the corresponding values in the first row')
        elif entry[0] == 'committor_analysis':
            if len(entry) > 3:
                sys.exit('Error: committor_analysis cannot contain whitespace (\' \') characters')
            committor_analysis = ast.literal_eval(entry[2])
            if committor_analysis:  # to allow committor_analysis = [], only perform these checks if it's not
                if not len(committor_analysis) == 5 and not len(committor_analysis) == 6:
                    sys.exit('Error: committor_analysis must be of length five or six (yours is of length ' + str(
                        len(committor_analysis)) + ')')
                elif len(committor_analysis) == 5:
                    committor_analysis.append('')  # if it's not supplied, use an empty string for committor_suffix
                for i in range(len(committor_analysis)):
                    if i in [1, 2] and type(committor_analysis[i]) not in [float, int]:
                        sys.exit('Error: committor_analysis[' + str(
                            i) + '] must have type float or int, but has type: ' + str(type(committor_analysis[i])))
                    elif i in [0, 3, 4] and type(committor_analysis[i]) not in [int]:
                        sys.exit('Error: committor_analysis[' + str(i) + '] must have type int, but has type: ' + str(
                            type(committor_analysis[i])))
                    elif i == 5 and type(committor_analysis[i]) not in [str]:
                        sys.exit(
                            'Error: committor_analysis[' + str(i) + '] must have type string, but has type: ' + str(
                                type(committor_analysis[i])))
        elif entry[0] == 'eps_settings':
            if len(entry) > 3:
                sys.exit('Error: eps_settings cannot contain whitespace (\' \') characters')
            eps_settings = ast.literal_eval(entry[2])
            if eps_settings:  # to allow eps_settings = [], only perform these checks if it's not
                if not len(eps_settings) == 6:
                    sys.exit(
                        'Error: eps_settings must be of length six (yours is of length ' + str(len(eps_settings)) + ')')
                for i in range(len(eps_settings)):
                    if i in [2, 3, 5] and type(eps_settings[i]) not in [float, int]:
                        sys.exit(
                            'Error: eps_settings[' + str(i) + '] must have type float or int, but has type: ' + str(
                                type(eps_settings[i])))
                    elif i in [0, 1, 4] and type(eps_settings[i]) not in [int]:
                        sys.exit('Error: eps_settings[' + str(i) + '] must have type int, but has type: ' + str(
                            type(eps_settings[i])))
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
                                                                               'Rounding down to ' + str(
                    int(entry[2])) + ' and proceeding.')
            groupfile = int(entry[2])
            groupfile_list = []  # initialize list of groupfiles
        elif entry[0] == 'groupfile_max_delay':
            try:
                if int(entry[2]) < 0:
                    sys.exit('Error: groupfile_max_delay must be greater than or equal to 0')
            except ValueError:
                sys.exit('Error: groupfile_max_delay must be an integer')
            if isinstance(ast.literal_eval(entry[2]), float):
                print('Warning: groupfile_max_delay was given as a float (' + entry[2] + '), but should be an integer. '
                                                                                         'Rounding down to ' + str(
                    int(entry[2])) + ' and proceeding.')
            groupfile_max_delay = int(entry[2])
        elif entry[0] == 'include_qdot':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: include_qdot must be either True or False')
            include_qdot = str2bool(entry[2])
        elif entry[0] == 'eps_dynamic_seed':
            eps_dynamic_seed = ast.literal_eval(entry[2])
            if type(eps_dynamic_seed) not in [list, int]:
                sys.exit('Error: eps_dynamic_seed must be either an integer or a list (yours is of type ' + str(
                    type(entry[2])) + ')')
            empty_windows = []  # initialize empty windows list to support dynamic seeding
        elif entry[0] == 'minmax_error_behavior':
            if not entry[2] in ['exit', 'skip', 'accept']:
                sys.exit(
                    'Error: minmax_error_behavior must be either exit, skip, or accept (you gave ' + entry[2] + ')')
            minmax_error_behavior = entry[2]
        elif entry[0] == 'zip_for_transfer':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: zip_for_transfer must be either True or False')
            zip_for_transfer = str2bool(entry[2])
        elif entry[0] == 'cv_analysis':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: cv_analysis must be either True or False')
            cv_analysis = str2bool(entry[2])
        elif entry[0] == 'skip_log':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: skip_log must be either True or False')
            skip_log = str2bool(entry[2])
        elif entry[0] == 'bootstrap_n':
            try:
                null = int(entry[2])
            except ValueError:
                sys.exit('Error: bootstrap_n must be an integer')
            bootstrap_n = int(entry[2])
            bootstrap_flag = False     # indicates whether bootstrapping has converged
            bootstrap_bookkeep = False          # bookkeeping for handle_bootstrap()
            bootstrap_jobids = []      # used to track bootstrapping jobs
        elif entry[0] == 'bootstrap_threshold':
            try:
                null = float(entry[2])
            except ValueError:
                sys.exit('Error: bootstrap_threshold must be a float')
            bootstrap_threshold = float(entry[2])
        elif entry[0] == 'find_ts':
            try:
                if int(entry[2]) < 0:
                    sys.exit('Error: find_ts must be greater than or equal to 0')
            except ValueError:
                sys.exit('Error: find_ts must be an integer')
            if isinstance(ast.literal_eval(entry[2]), float):
                print('Warning: find_ts was given as a float (' + entry[2] + '), but should be an integer. '
                      'Rounding down to ' + str(int(entry[2])) + ' and proceeding.')
            if isinstance(ast.literal_eval(entry[2]), bool):
                print('Warning: find_ts was given as a bool (' + entry[2] + '), but should be an integer. '
                      'Interpreting as ' + str(int(entry[2])) + ' and proceeding.')
            find_ts = int(entry[2])
        elif entry[0] == 'as_out':
            as_out = entry[2]
        elif entry[0] == 'cleanup':
            if not entry[2].lower() == 'true' and not entry[2].lower() == 'false':
                sys.exit('Error: cleanup must be either True or False')
            cleanup = str2bool(entry[2])

    # Remove trailing '/' from working_directory and home_folder for compatibility with my code
    if working_directory[-1] == '/':
        working_directory = working_directory[:-1]
    if home_folder[-1] == '/':
        home_folder = home_folder[:-1]

    if not rc_minmax:
        if not os.path.exists(as_out):
            if os.path.exists(home_folder + '/' + as_out):
                as_out = home_folder + '/' + as_out
            elif os.path.exists(working_directory + '/' + as_out):
                as_out = working_directory + '/' + as_out
            elif committor_analysis or eps_settings or rc_definition:
                sys.exit('Error: rc_definition, committor_analysis and/or eps_settings options were given, but '
                         'rc_minmax was not and as_out file: ' + as_out + ' was not found when interpreted as an '
                         'absolute path, in the working directory ' + working_directory + ', or in the home folder '
                         + home_folder + '. Provide a valid entry for either as_out or rc_minmax.')
        if committor_analysis or eps_settings or rc_definition:  # in this case, we need to build rc_minmax from as_out
            rc_minmax = [[],[]]
            asoutlines = [[float(item) for item in line.replace('A <- ', '').replace('B <- ', '').replace(' \n', '').replace('\n', '').split(' ')] for line in open(as_out, 'r').readlines()]
            open(as_out, 'r').close()
            mapped = list(map(list, zip(*asoutlines)))
            rc_minmax = [[numpy.min(item) for item in mapped], [numpy.max(item) for item in mapped]]

    # Initialize jinja2 environment for filling out templates
    if os.path.exists(home_folder + '/' + 'templates'):
        env = Environment(
            loader=FileSystemLoader(home_folder + '/' + 'templates'),
        )
    else:
        sys.exit('Error: could not locate templates folder: ' + home_folder + '/' + 'templates\nSee documentation for '
                                                                                    'the \'home_folder\' option.')

    # Update global settings Namespace object to store all these variables
    settings.__dict__.update(locals())

    # Call rc_eval if we're doing an rc_definition or committor_analysis run
    if rc_definition and not (committor_analysis or eps_settings):
        rc_eval.return_rcs(settings)
        sys.exit(
            '\nCompleted reaction coordinate evaluation run. See ' + working_directory + '/rc_eval.out for results.')
    elif rc_definition and committor_analysis:
        rc_eval.committor_analysis(settings)
        sys.exit(
            '\nCompleted committor analysis run. See ' + working_directory + '/committor_analysis' + committor_analysis[
                5] + '/committor_analysis.out for results.')
    elif not rc_definition and committor_analysis:
        sys.exit('Error: committor analysis run requires rc_definition to be defined')

    # Make the working directory if it doesn't already exist and should be made
    dirName = working_directory
    if not resample and not restart and not zip_for_transfer and not cv_analysis and overwrite:  # if (resample or restart or zip_for_transfer or cv_analysis) == True, we want to keep our old working directory
        if os.path.exists(dirName):
            shutil.rmtree(dirName)  # delete old working directory
        os.makedirs(dirName)  # make a new one
    elif resample or restart or zip_for_transfer or cv_analysis:
        if not os.path.exists(dirName):
            whichone = ''
            if resample:
                whichone = 'resample'
            elif restart:
                whichone = 'restart'
            elif zip_for_transfer:
                whichone = 'zip_for_transfer'
            elif cv_analysis:
                whichone = 'cv_analysis'
            sys.exit('Error: ' + whichone + ' = True, but I can\'t find the working directory: ' + dirName)
    elif not overwrite:  # if (resample or restart or zip_for_transfer or cv_analysis) == False and overwrite == False, make sure dirName doesn't exist...
        if os.path.exists(dirName):
            sys.exit('Error: overwrite = False, but working directory ' + dirName + ' already exists. Move it, choose a'
                     ' different working directory, or add option -O to overwrite it.')
        else:
            os.makedirs(dirName)

    if not resample and not restart and not zip_for_transfer and not cv_analysis and not find_ts:
        if if_glob:
            start_name = glob.glob(initial_structure)  # list of names of coordinate files to begin shooting from
        else:
            start_name = [initial_structure]

        for filename in start_name:
            if ' ' in filename:
                sys.exit('Error: one or more input coordinate filenames contains a space character (\' \'), which is '
                         'not supported. The first offending filename found was: ' + filename)
            elif 'bootstrap' in filename:
                sys.exit('Error: the string \'bootstrap\' is reserved for internal use and cannot be included in part '
                         'of any input coordinate filename. The first offending filename found was: ' + filename)

        if len(start_name) == 0:
            sys.exit('Error: no initial structure found. Check input options initial_structure and if_glob.')
        for init in start_name:
            if not os.path.exists(init):
                sys.exit('Error: could not find initial structure file: ' + init)

        if degeneracy > 1:
            temp = []  # initialize temporary list to substitute for start_name later
            for init in start_name:
                for i in range(degeneracy):
                    shutil.copy(init, init + '_' + str(i + 1))
                    temp.append(init + '_' + str(i + 1))
            start_name = temp
    elif find_ts:   # in this case, want to build start_name using the init_find_ts() function.
        # First, make sure there are no mutually exclusive options provided...
        if resample:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and resample = True')
        if rc_definition:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and rc_definition is defined')
        if eps_settings:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and eps_settings is defined')
        if committor_analysis:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and committor_analysis is defined')
        if zip_for_transfer:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and zip_for_transfer = True')
        if cv_analysis:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and cv_analysis = True')
        if restart:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and restart = True')
        if degeneracy > 1:
            sys.exit('Error: the following options are incompatible: find_ts > 0 and degeneracy > 1')
        # Then grab the input structure and throw an error if there's more than one
        if if_glob:
            start_name = glob.glob(initial_structure)  # list of names of coordinate files to begin shooting from
        else:
            start_name = [initial_structure]
        if len(start_name) > 1:
            sys.exit('Error: only one initial structure is permitted when find_ts > 0. The combination of your '
                     'initial_structure and if_glob settings matches ' + str(len(start_name)) + ' coordinate files. '
                     'If you want to build transition state guesses from multiple initial coordinate files, please '
                     'perform a separate ATESA run for each one.')
        os.chdir(working_directory)                                 # move to working directory
        try:
            os.remove(settings.logfile)     # delete previous run's log (find_ts is always a new ATESA job)
        except OSError:                     # catches error if no previous log file exists
            pass
        with open(settings.logfile, 'w+') as newlog:
            localtime = time.localtime()
            mins = str(localtime.tm_min)
            if len(mins) == 1:
                mins = '0' + mins
            secs = str(localtime.tm_sec)
            if len(secs) == 1:
                secs = '0' + secs
            newlog.write('~~~New log file ' + str(localtime.tm_year) + '-' + str(localtime.tm_mon) + '-' +
                         str(localtime.tm_mday) + ' ' + str(localtime.tm_hour) + ':' + mins + ':' + secs + '~~~')
            newlog.close()
        for structure in start_name:                                # for all of the initial structures (should be one)
            shutil.copy(called_path + '/' + structure, './')        # copy to the working directory...
            try:
                shutil.copy(called_path + '/' + topology, './')     # ... and its little topology file, too!
            except OSError:
                sys.exit('Error: could not find the indicated topology file: ' + called_path + '/' + topology)
        start_name = init_find_ts(settings,start_name)

    os.chdir(working_directory)  # move to working directory

    # Initialize files needed for an EPS run if that's what we're doing.
    # Have to do this before potentially calling main_loop() for a restart run if we want that to work.
    if eps_settings:
        initialize_eps(settings)

    # Check for incompatible options and return helpful error message if a conflict is found
    if resample or committor_analysis or (rc_definition and not eps_settings) or zip_for_transfer or cv_analysis:
        problem = '[UNDEFINED (developer error, please raise GitHub issue)]'
        if resample:
            problem = 'resample = True'
        elif (rc_definition and not eps_settings):
            problem = 'rc_definition is defined and eps_settings is not'
        elif committor_analysis:
            problem = 'committor_analysis is defined'
        elif zip_for_transfer:
            problem = 'zip_for_transfer = True'
        elif cv_analysis:
            problem = 'cv_analysis = True'
        if restart:  # todo: this is crazy ugly...
            sys.exit('Error: the following options are incompatible: restart = True and ' + problem)
        try:
            open(settings.logfile, 'r').close()
        except (IOError, OSError):
            if skip_log == True or eps_settings:
                pass
            else:
                sys.exit('Error: ' + problem + ' but I cannot open ' + logfile + ' in the working directory: ' + working_directory)
    if fork > 1 and eps_settings:
        sys.exit('Error: options \'fork\ > 1\' and \'eps_settings\' are incompatible')

    # Run auxiliary functions if appropriate
    if zip_for_transfer:
        # Load in pickle and move directly to zip_for_transfer_func()
        try:
            settings.allthreads = pickle.load(open('restart.pkl', 'rb'))
            settings.restartthreads = settings.allthreads
        except (IOError, OSError):
            sys.exit(
                'Error: zip_for_transfer = True, but I cannot read restart.pkl inside working directory: ' + working_directory)
        zip_for_transfer_func(settings)
        sys.exit('Zipped up files in: ' + working_directory + '/transfer.zip. Exiting normally.')
    if cv_analysis:
        cv_analysis_func(settings)
        sys.exit('CV analysis complete, files in ' + working_directory + '/cv_analysis/')

    # Handle initialization of restart settings if appropriate
    if restart:
        # First, carefully load in the necessary information:
        # try:
        #     os.chdir(working_directory)
        # except (IOError, OSError):
        #     sys.exit('Error: restart = True, but I cannot find the working directory: ' + working_directory)
        try:
            settings.allthreads = pickle.load(open('restart.pkl', 'rb'))
            settings.restartthreads = settings.allthreads
        except (IOError, OSError):
            sys.exit(
                'Error: restart = True, but I cannot read restart.pkl inside working directory: ' + working_directory)
        # If we're restarting an EPS run, we need to ensure that the EPS settings are compatible. Specifically, we need
        # to ensure that the window boundaries that were previously used still exist in the new ones, and that the
        # number of beads per thread is the same. We would also like to check that the traj_length is the same ideally,
        # but this information is not stored in threads.
        if eps_settings:
            for thread in settings.allthreads:
                if ('%.3f' % (thread.rc_min + settings.overlap) not in ['%.3f' % x for x in settings.eps_windows]) or (
                        '%.3f' % (thread.rc_max - settings.overlap) not in ['%.3f' % x for x in settings.eps_windows]):
                    if '%.3f' % (thread.rc_min + settings.overlap) not in ['%.3f' % (x) for x in settings.eps_windows]:
                        offending = str(thread.rc_min + settings.overlap)
                    else:
                        offending = str(thread.rc_max - settings.overlap)
                    sys.exit('Error: attempted to restart this EPS run, but the restart.pkl file contained threads with'
                             ' different reaction coordinate boundaries than those defined by the eps_settings option '
                             'for this job. The offending cutoff value (including overlap) was: ' + offending)
                elif not thread.eps_fwd + thread.eps_bwd + 1 == settings.k_beads:
                    sys.exit('Error: attempted to restart this EPS run, but the restart.pkl file contained threads with'
                             ' a different number of beads than specified by the eps_settings option for this job. '
                             'The restart.pkl threads contain ' + str(thread.eps_fwd + thread.eps_bwd + 1) + ' beads, '
                             'whereas for this run k_beads was set to: ' + str(settings.k_beads))
            if settings.eps_dynamic_seed:
                # Need to handle empty_windows, since unlike in normal behavior threads are beginning (restarting)
                # without going through spawnthread().
                if type(eps_dynamic_seed) == int:
                    settings.eps_dynamic_seed = [eps_dynamic_seed for null in range(len(settings.eps_windows) - 1)]
                elif not len(eps_dynamic_seed) == (len(settings.eps_windows) - 1):
                    sys.exit('Error: eps_dynamic_seed was given as a list, but is not of the same length as the '
                             'number of EPS windows. There are ' + str((len(settings.eps_windows) - 1)) + ' EPS windows'
                             ' but eps_dynamic_seed is of length ' + str(len(settings.eps_dynamic_seed)))
                window_index = 0
                for window in range(len(settings.eps_windows) - 1):
                    settings.empty_windows.append(settings.eps_dynamic_seed[window_index])  # meaning eps_dynamic_seed[window_index] more threads need to start here before it will no longer be considered "empty"
                    window_index += 1
                for thread in settings.allthreads:
                    if thread.status not in ['max_accept', 'max_moves', 'max_fails']:
                        window_index = ['%.3f' % x for x in settings.eps_windows].index('%.3f' % (thread.rc_min + settings.overlap))
                        settings.empty_windows[window_index] -= 1
                        if settings.empty_windows[window_index] < 0:
                            settings.empty_windows[window_index] = 0
                            settings.restartthreads.remove(thread)  # so as not to restart more threads than requested

        settings.running = []
        settings.itinerary = []
        # Next, add those threads that haven't terminated to the itinerary and call main_loop
        for thread in settings.restartthreads:
            if thread.status not in ['max_accept', 'max_moves', 'max_fails']:
                settings.itinerary.append(thread)
                if not eps_settings and thread.type == 'prod':  # remove old output trajs to prevent crossed wires
                    try:
                        os.remove(working_directory + '/' + thread.name + '_fwd.nc')
                    except (IOError, OSError):   # old file never written; not a problem!
                        pass
                    try:
                        os.remove(working_directory + '/' + thread.name + '_bwd.nc')
                    except (IOError, OSError):
                        pass
                    thread.commit1 = ''
                    thread.commit2 = ''
        localtime = time.localtime()
        mins = str(localtime.tm_min)
        if len(mins) == 1:
            mins = '0' + mins
        secs = str(localtime.tm_sec)
        if len(secs) == 1:
            secs = '0' + secs
        open(settings.logfile, 'a').write(
            '\n~~~Restarting ' + str(localtime.tm_year) + '-' + str(localtime.tm_mon) + '-' +
            str(localtime.tm_mday) + ' ' + str(localtime.tm_hour) + ':' + mins + ':' + secs + '~~~')
        main_loop(settings)
        sys.exit()  # not an error, this is exiting normally

    if eps_settings and (always_new or bootstrap_n):
        if always_new:
            sys.exit('Error: the following options are incompatible: eps_settings and always_new')
        if bootstrap_n:
            sys.exit('Error: the following options are incompatible: eps_settings and bootstrap_n')

    # Check if candidate_cv is given explicitly or as [mask, coordinates], and if the latter, build the explicit form
    # NOTE: This code works and remains in place here for posterity; however, actually using it is probably not advised,
    # as it usually produces far too many OPs to practically test with LMAX.
    if not isinstance(candidateops[0], list) and not literal_ops:
        try:
            traj = pytraj.iterload(candidateops[1], topology)
        except RuntimeError:
            sys.exit('Error: unable to load coordinate file ' + candidateops[1] + ' with topology ' + topology)
        traj.top.set_reference(traj[0])  # set reference frame to first frame
        atom_indices = list(pytraj.select(candidateops[0], traj.top))
        if not atom_indices:  # if distance = 0 or if there's a formatting error
            sys.exit('Error: found no atoms matching ' + candidateops[0] + ' in file ' + candidateops[1]
                     + '. This may indicate an issue with the format of the candidate_cv option!')
        elif len(atom_indices) == 1:  # if formatted correctly but only matches self
            sys.exit('Error: mask ' + candidateops[0] + ' with file ' + candidateops[1] + ' only matches one atom.'
                     + ' Cannot produce any order parameters!')

        # Super-ugly nested loops to build every combination of indices...
        temp_ops = [[], [], [], []]  # temporary list to append candidate ops to as they're built
        count = 0
        for index in atom_indices:
            count += 1
            update_progress(count / len(atom_indices), 'Building all possible order parameters using atoms matching ' +
                            candidateops[1] + ' with file ' + candidateops[0])
            for second_index in atom_indices:  # second-order connections (distances)
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
        mirrors_list = []  # list containing mirrors of OP definitions
        # If an OP is already in mirrors_list, it's deleted from temp_ops; if not, its mirror is added to temp_ops
        while position < len(temp_ops[0]):
            count += 1
            update_progress(count / total_iters, 'Removing redundant coordinates')
            one = temp_ops[0][position]
            two = temp_ops[1][position]
            three = temp_ops[2][position]
            four = temp_ops[3][position]
            if (four and [one, two, three, four] in mirrors_list) \
                    or (three and not four and [one, two, three] in mirrors_list) \
                    or (not three and not four and [one, two] in mirrors_list):
                del temp_ops[0][position]
                del temp_ops[1][position]
                del temp_ops[2][position]
                del temp_ops[3][position]
                position -= 1
            elif four:
                mirrors_list.append([four, three, two, one])
            elif three:
                mirrors_list.append([three, two, one])
            else:
                mirrors_list.append([two, one])
            position += 1

        settings.candidateops = temp_ops
        print('\nThe finalized order parameter definition is: ' + str(settings.candidateops))

    # Return an error and exit if the input file is missing entries for non-optional inputs.
    if 'commit_fwd' not in [entry[0] for entry in input_file_lines] and not (resample or rc_definition or eps_settings):
        sys.exit('Error: input file is missing entry for commit_fwd, which is non-optional')
    if 'commit_bwd' not in [entry[0] for entry in input_file_lines] and not (resample or rc_definition or eps_settings):
        sys.exit('Error: input file is missing entry for commit_bwd, which is non-optional')
    if 'candidate_cv' not in [entry[0] for entry in input_file_lines] and 'candidate_op' not in [entry[0] for entry in input_file_lines]:
        sys.exit('Error: input file is missing entry for candidate_cv, which is non-optional')

    if not resample and not restart:
        settings.itinerary = []     # a list of threads that need running
        settings.running = []       # a list of currently running threads
        settings.allthreads = []    # a list of all threads regardless of status

        if not find_ts: # if find_ts, this has already been done
            try:
                os.remove(settings.logfile)  # delete previous run's log
            except OSError:  # catches error if no previous log file exists
                pass
            with open(settings.logfile, 'w+') as newlog:
                localtime = time.localtime()
                mins = str(localtime.tm_min)
                if len(mins) == 1:
                    mins = '0' + mins
                secs = str(localtime.tm_sec)
                if len(secs) == 1:
                    secs = '0' + secs
                newlog.write('~~~New log file ' + str(localtime.tm_year) + '-' + str(localtime.tm_mon) + '-' +
                             str(localtime.tm_mday) + ' ' + str(localtime.tm_hour) + ':' + mins + ':' + secs + '~~~')
                newlog.close()

        for structure in start_name:    # for all of the initial structures...
            if not find_ts:             # if find_ts, structures are already in working directory
                shutil.copy(called_path + '/' + structure, './')  # copy the input structure to the working directory...
                try:
                    shutil.copy(called_path + '/' + topology, './')  # ... and its little topology file, too!
                except OSError:
                    sys.exit('Error: could not find the indicated topology file: ' + called_path + '/' + topology)
            thread = spawnthread(structure, suffix='1', settings=settings)  # spawn a new thread with default settings
            # thread.last_valid = '0'                                 # so that if the first shooting point does not result in a valid transition path, shooting will begin from the TS guess
            thread.prmtop = topology  # set prmtop filename for the thread
            settings.itinerary.append(thread)  # submit it to the itinerary
            if degeneracy > 1:  # if degeneracy > 1, files in start_name were copies...
                os.remove(called_path + '/' + structure)  # ... delete them to keep the user's space clean!

    try:
        os.remove('as.out')  # delete previous run's output file
    except OSError:  # catches error if no previous output file exists
        pass
    if not eps_settings:
        with open('as.out', 'w+') as newout:  # make a new output file
            newout.close()

    # Implementation of resample
    if resample and not eps_settings:  # if True, this is a resample run, so we'll head off the simulations steps here
        pattern = re.compile('\ .*\ finished')  # pattern to find job name
        pattern2 = re.compile('result:\ [a-z]*\ ')  # pattern for basin commitment flag
        if not skip_log:
            try:
                this_logfile = open(logfile)  # open log for reading...
            except OSError:
                sys.exit('Error: could not find ' + logfile + ' in working directory: ' + working_directory)
            logfile_lines = this_logfile.readlines()
            this_logfile.close()
            count = 0
            for line in logfile_lines:  # iterate through log file
                if 'finished with fwd trajectory result: ' in line:  # looking for lines with results
                    commit = pattern2.findall(line)[0][8:-1]  # first, identify the commitment flag
                    if commit != 'fail':  # none of this matters if it was "fail"
                        basin = 'error'
                        if commit == 'fwd':
                            basin = 'B'
                        elif commit == 'bwd':
                            basin = 'A'
                        init_name = pattern.findall(line)[0][5:-9] + '_init_fwd.rst'  # clunky; removes "run " and " finished"
                        open('as.out', 'a').write(basin + ' <- ' + candidatevalues(init_name, settings=settings) + '\n')
                        open(settings.logfile, 'a').close()
                count += 1
                update_progress(count / len(logfile_lines), 'Resampling by searching through logfile')
        else:  # we can't use the log file, so we'll use the thread.history attributes instead
            try:  # I need allthreads to identify threads with zero accepted moves and skip them
                allthreads = pickle.load(open('restart.pkl', 'rb'))
            except (IOError, OSError):
                sys.exit(
                    'Error: (resample and skip_log) = True, but I cannot read restart.pkl inside working directory: ' + working_directory)
            num_moves = 0
            for thread in allthreads:
                num_moves += len(thread.history)
            count = 0
            update_progress(count / num_moves, 'Resampling by searching through thread histories')
            for thread in allthreads:
                before_first_accept = True
                for move in thread.history:
                    count += 1
                    if move[-1] not in ['F', 'B', 'S', 'X']:  # just a brief sanity check
                        sys.exit('Error: thread history for thread: ' + thread.basename + ' is formatted incorrectly')

                    if before_first_accept:
                        if move[-1] == 'S':
                            before_first_accept = False
                            fwd_commit = standalone_checkcommit(move[:-2] + '_fwd.nc', settings=settings)
                            if fwd_commit == 'fwd':
                                fwd_commit = 'B'
                            elif fwd_commit == 'bwd':
                                fwd_commit = 'A'
                                # Else, fwd_commit = 'fail'
                    elif move[-1] == 'F':  # fwd, commit to 'B'
                        fwd_commit = 'B'
                    elif move[-1] == 'B':  # bwd, commit to 'A'
                        fwd_commit = 'A'
                    elif move[-1] == 'S':  # have to check, in this case
                        fwd_commit = standalone_checkcommit(move[:-2] + '_fwd.nc', settings=settings)
                        if fwd_commit == 'fwd':
                            fwd_commit = 'B'
                        elif fwd_commit == 'bwd':
                            fwd_commit = 'A'
                            # Else, fwd_commit = 'fail'

                    if not before_first_accept and not move[-1] == 'X' and not fwd_commit == 'fail':
                        open('as.out', 'a').write(fwd_commit + ' <- ' + candidatevalues(move[:-2] + '_init_fwd.rst',
                                                                                        settings=settings) + '\n')
                        open(settings.logfile, 'a').close()

                    update_progress(count / num_moves, 'Resampling by searching through thread histories')
        sys.exit('Resampling complete; wrote new output to ' + working_directory + '/as.out')
    elif resample and eps_settings:
        def report_rc_values(coord_file, thread):
            # Simple function for outputting the RC values for a given trajectory traj to the eps_results.out file
            rc_values = []
            if '.rst' in coord_file or '.rst7' in coord_file:
                fileformat = '.rst7'
            elif '.nc' in coord_file:
                fileformat = '.nc'
            else:
                sys.exit('Error: report_rc_values() encountered a file of unknown format: ' + coord_file)
            traj = pytraj.iterload(coord_file, thread.prmtop, format=fileformat)
            for i in range(traj.__len__()):  # iterate through frames of traj
                cv_values = [float(cv) for cv in candidatevalues(coord_file, frame=i, reduce=True, settings=settings).split(' ') if cv]  # CV values as a list
                rc_values.append(get_rc_value(cv_values=cv_values, settings=settings))
            for value in rc_values:
                if thread.rc_min <= value <= thread.rc_max:  # only write to output if the bead is inside the window
                    open('eps_results.out', 'a').write(str(thread.rc_min) + ' ' + str(thread.rc_max) + ' ' + str(value) + '\n')
                    open('eps_results.out', 'a').close()
            return rc_values

        try:  # I need allthreads to identify threads with zero accepted moves and skip them
            allthreads = pickle.load(open('restart.pkl', 'rb'))
        except (IOError, OSError):
            sys.exit(
                'Error: (resample and eps_settings) = True, but I cannot read restart.pkl inside working directory: ' + working_directory)
        num_moves = 0
        for thread in allthreads:
            num_moves += len(thread.history)
        count = 0
        update_progress(count / num_moves, 'Resampling by searching through thread histories')
        for thread in allthreads:
            thread.rc_min = ''
            thread.rc_max = ''
            for move in thread.history:
                count += 1
                if move[-1] not in ['F', 'B', 'S', 'X']:  # just a brief sanity check
                    sys.exit('Error: thread history for thread: ' + thread.basename + ' is formatted incorrectly')

                if thread.rc_min == '':
                    cv_values = [float(cv) for cv in candidatevalues(move.split(' ')[0] + '_init_fwd.rst', reduce=True, settings=settings).split(' ') if cv]  # CV values as a list
                    rc_value = get_rc_value(cv_values=cv_values, settings=settings)
                    for window_index in range(len(settings.eps_windows) - 1):
                        if settings.eps_windows[window_index] <= rc_value <= settings.eps_windows[window_index + 1]:
                            thread.rc_min = settings.eps_windows[window_index] - settings.overlap
                            thread.rc_max = settings.eps_windows[window_index + 1] + settings.overlap
                            break

                report_rc_values(move.split(' ')[0] + '_init_fwd.rst', thread)
                if os.path.exists(move.split(' ')[0] + '_fwd.nc'):
                    report_rc_values(move.split(' ')[0] + '_fwd.nc', thread)
                if os.path.exists(move.split(' ')[0] + '_bwd.nc'):
                    report_rc_values(move.split(' ')[0] + '_bwd.nc', thread)

                update_progress(count / num_moves, 'Resampling by searching through EPS thread histories')
        sys.exit('Resampling complete; wrote new output to ' + working_directory + '/eps_results.out')

    os.makedirs('history')  # make a new directory to contain the history files of each thread

    main_loop(settings)     # this call to main_loop only reached if this is a new run (not restarted)
    sys.exit()              # not an error, this is exiting normally


# Define runtime for when this program is called
if __name__ == '__main__':
    settings = argparse.Namespace()
    main()  # doesn't return a status and is not followed by sys.exit() because main() and subfunctions handle that
