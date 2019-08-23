# Script to test my bootstrapping code on pre-generated as.out data, rather than integrated with atesa. Includes a
# modified handle_bootstrap() function definition, followed by definitions of helper functions and then by some quick
# code to feed it fractions of as.out

from __future__ import division     # causes division to work properly when using Python 2
import os
import pytraj
import glob
import numpy                        # required to support numpy functions passed in rc_definition
import sys
import re
import importlib
import subprocess
import time
import math
import argparse
import random
import shutil
from jinja2 import Environment, FileSystemLoader

# Setup settings object
if True:
    initial_structure = 'inpcrd'  # Initial structure filename
    if_glob = False  # True if initial_structure should be interpreted as a glob argument
    topology = 'test.prmtop'  # Topology filename
    n_adjust = 50  # Max number of frames by which each step deviates from the previous one
    batch_system = 'slurm'  # Batch system type, Slurm or PBS
    working_directory = os.path.dirname(os.path.realpath(__file__))  # Working directory for aimless shooting calculations
    restart_on_crash = False  # If a thread crashes during initialization, should it be resubmitted?
    max_fails = -1  # Number of unaccepted shooting moves before a thread is killed; a negative value means "no max"
    max_moves = -1  # Number of moves with any result permitted before the thread is terminated
    max_accept = -1  # Number of accepted moves permitted before the thread is terminated
    degeneracy = 1  # Number of duplicate threads to produce for each initial structure
    init_nodes = 1  # Number of nodes on which to run initialization jobs
    init_ppn = 1  # Number of processors per node on which to run initialization jobs
    init_walltime = '01:00:00'  # Wall time for initialization jobs
    init_mem = '4000mb'  # Memory per core for initialization jobs
    prod_nodes = 1  # Number of nodes on which to run production jobs
    prod_ppn = 1  # Number of processors per node on which to run production jobs
    prod_walltime = '01:00:00'  # Wall time for production jobs
    prod_mem = '4000mb'  # Memory per core for intialization jobs
    resample = False  # If True, don't run any new simulations; just rewrite as.out based on current settings and existing data
    fork = 1  # Number of new threads to spawn after each successful shooting move. Each one gets its own call to pickframe()
    home_folder = os.path.dirname(os.path.realpath(__file__))  # Directory containing templates and input_files folders
    always_new = False  # Pick a new shooting move after every move, even if it isn't accepted?
    committor_analysis = ''  # Variables to pass into committor analysis.
    restart = False  # Whether or not to restart an old AS run located in working_directory
    groupfile = 0  # Number of jobs to submit in one groupfile, if necessary
    groupfile_max_delay = 3600  # Time in seconds to allow groupfiles to remain in construction before submitting
    eps_settings = []  # Equilibrium Path Sampling: [n_windows, k_beads, rc_min, rc_max, traj_length, overlap]
    eps_dynamic_seed = ''  # Flag to seed empty windows using beads from other trajectories during EPS
    minmax_error_behavior = 'exit'  # Indicates behavior when an reduced OP value is outside range [0,1]
    zip_for_transfer = False  # Indicates that we should simply zip up necessary files and quit
    cv_analysis = False  # Indicates that we just want to produce a CV analysis output folder
    skip_log = False  # Indicates that the user knows the log file is missing or broken and can't be used for analysis scripts
    DEBUGMODE = False
    bootstrap_threshold = 0.1  # Threshold for comparison of model RC coefficients during RC bootstrapping
    cvs_in_rc = 3  # Number of CVs to include in RCs produced in the course of sampling
    find_ts = 0
    # From here down not actually defaults for atesa.py, but set for expedience of testing
    bootstrap_n = 5  # Number of bootstrapped models to build during RC bootstrapping
    bootstrap_flag = False  # indicates whether bootstrapping has converged
    bootstrap_bookkeep = True  # bookkeeping for handle_bootstrap()
    bootstrap_jobids = []  # used to track bootstrapping jobs
    logfile = 'test.log'
    committor_analysis_options = ''
    n_shots = 0
    include_qdot = False
    candidateops = ['pytraj.distance(traj,mask=\'@4272 @7175\')[0]',
                    'pytraj.distance(traj,mask=\'@7175 @7174\')[0]']
    rc_definition = '2*OP1 - OP2'
    rc_minmax = [[1, 0.6], [3, 2]]
    literal_ops = True  # since in ATESA this is set in initialization
    itinerary = []
    running = []
    allthreads = []
    commit_define_fwd = [['@7185', '@7185'], ['@7174', '@7186'], [1.60, 2.75], ['lt', 'gt']]
    commit_define_bwd = [['@7185', '@7185'], ['@7174', '@7186'], [2.75, 2.00], ['gt', 'lt']]

    if os.path.exists(home_folder + '/' + 'templates'):
        env = Environment(
            loader=FileSystemLoader(home_folder + '/' + 'templates'),
        )
    else:
        sys.exit('Error: could not locate templates folder: ' + home_folder + '/' + 'templates\nSee documentation for '
                                                                                    'the \'home_folder\' option.')

    settings = argparse.Namespace()
    settings.__dict__.update(locals())

# Define functions and Thread class
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
                        return 'Converged but CV absent from reference'  # CV absent from reference
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
                    return 'Converged but coeff outside reference range'  # coefficient outside of reference range
        return True  # Only happens if every other check above was passed

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
        open(settings.logfile, 'a').write('\nBatch system says: ' + str(output)) # todo: put this inside interact? As written, this always prints just the jobid of the submitted job rather than the full message.
        open(settings.logfile, 'a').close()
        return output


def interact(type,settings):
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
    # todo: replace system-specific error handling here with more general solution
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
            return pattern.findall(str(output))[0]
        except IndexError:
            sys.exit('Error: unable to submit a batch job: ' + str(type) + '. Got message: ' + str(output))

# Finally, handle cutting up of as.out, calling of handle_bootstrap, and interpretation of results.
# First step is to write a new file called "as.out" containing the desired fraction of the full file.
shutil.copy('as.out','as_full.out')
this_len = 1176
full_len = len(open('as_full.out', 'r').readlines())
while this_len <= full_len:
    this_len += int(full_len/10)
    open('as.out','w').write('')
    for item in open('as_full.out', 'r').readlines()[:this_len]:
        open('as.out','a').write(item)
    open('as.out', 'a').close()
    open('as_full.out', 'r').close()
    result = False
    settings.len_data = this_len
    while result == False:
        result = handle_bootstrap(settings)
        time.sleep(60)
