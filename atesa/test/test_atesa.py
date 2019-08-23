import pytest
import filecmp
from atesa import *
import atesa

def default_globals():
    # Assembles default global variables and returns them as a dictionary.
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
    bootstrap_n = 5  # Number of bootstrapped models to build during RC bootstrapping
    cvs_in_rc = 3  # Number of CVs to include in RCs produced in the course of sampling
    find_ts = 0
    cleanup = False
    # From here down not actually defaults for atesa.py, but set for expedience of testing
    logfile = 'test.log'
    committor_analysis_options = ''
    n_shots = 0
    include_qdot = False
    candidateops = ['pytraj.distance(traj,mask=\'@4272 @7175\')[0]',
                    'pytraj.distance(traj,mask=\'@7175 @7174\')[0]']
    rc_definition = '2*CV1 - CV2'
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

    return settings


def my_file_cmp(file1, file2):
    # Just a quick script to iteratively compare two files line-by-line
    # Using this instead of filecmp.cmp() to avoid recursive depth issues
    local_index = -1
    lines2 = open(file1, 'rb').readlines()
    for line1 in open(file1, 'rb').readlines():
        local_index += 1
        line2 = lines2[local_index]
        if not line1 == line2:
            print('File comparison error between files ' + file1 + ' and ' + file2 + ':\n ' + file1 + ': ' + line1 +
                  '\n ' + file2 + ': '+ line2)
            return False
    return True


def return_fake_slurm_squeue_output():
    return 'JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n\
         20329645   compute 1.1_2.1_  tburgin CG       0:02      1 comet-04-70\n\
         20329646   compute 1.1_2.1_  tburgin PD       0:00      1 (None)\n\
         20329647   compute 1.1_2.1_  tburgin PD       0:00      1 (None)\n\
         20329648   compute 1.1_2.1_  tburgin  R       2:25      1 comet-04-64\n\
         20329649   compute 1.1_2.1_  tburgin  R       1:27      1 comet-13-11'


def test_handle_groupfile():
    # todo: implement this eventually
    return ''   # to preclude rest of test
    groupfile_list = []
    result = handle_groupfile('test_job')
    assert result == 'groupfile_1'


def test_spawnthread_bare():
    settings = default_globals()
    this_thread = atesa.spawnthread('fakethread', thread_type='init', suffix='1', settings=settings)
    assert this_thread.basename == 'fakethread'
    assert this_thread.suffix == '1'
    assert this_thread.name == 'fakethread_1'
    assert this_thread.type == 'init'
    assert this_thread.start_name == 'fakethread'
    assert this_thread.prmtop == 'test.prmtop'
    assert this_thread.last_valid == '0'
    return this_thread


def test_initialize_eps():
    settings = default_globals()
    settings.eps_settings = [20, 10, -4, 4, 20, 0.1]
    atesa.initialize_eps(settings)  # updates settings because python is pass-by-object-reference
    assert [float('%.3f' % val) for val in settings.eps_windows] == [-4.0,-3.6,-3.2,-2.8,-2.4,-2.0,-1.6,-1.2,-0.8,-0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
    assert my_file_cmp(settings.working_directory + '/test_eps_results.out', settings.working_directory + '/eps_results.out')
    os.remove(settings.working_directory + '/eps_results.out')
    return settings     # have to return it because test_spawnthread_eps doesn't pass settings to this function


def test_spawnthread_eps():
    # Doesn't test basic spawnthread functions (test_spawnthread_bare() does); this tests EPS functionality
    # Spawnthread will call atesa.candidatevalues() to assemble CV values, so I need to give it real files to work with
    settings = test_initialize_eps()
    this_thread = atesa.spawnthread('eps.rst7', thread_type='init', suffix='1', settings=settings)
    assert float('%.3f' % (this_thread.rc_min)) == 0.7
    assert float('%.3f' % (this_thread.rc_max)) == 1.3
    assert this_thread.eps_bwd + this_thread.eps_fwd == settings.eps_settings[1] - 1    # -1 to account for start bead
    for i in range(settings.k_beads - 1):
        os.remove(settings.working_directory + '/eps' + str(i+1) + '.in')
    os.remove(settings.logfile)
    return this_thread


def test_makebatch_init():
    # This test (and the other makebatch tests) function as follows:
    #  Assemble spoofed **kwargs input to direct makebatch behavior
    #  Call makebatch with spoofed input
    #  Compare produced batch file to correct one in test directory w/ assert
    #  Delete new file to clean up
    settings = default_globals()
    this_thread = atesa.spawnthread('fakethread',thread_type='init',suffix='1',settings=settings)
    test_file = atesa.makebatch(this_thread,settings=settings)
    assert my_file_cmp(settings.working_directory + '/' + test_file, settings.working_directory + '/test_init.slurm')    # todo: this (and other filecmp's) rely on being localized on my particular filesystem... possible workaround?
    os.remove(test_file)
    os.remove(settings.logfile)


def test_makebatch_prod():
    settings = default_globals()
    this_thread = atesa.spawnthread('fakethread',thread_type='prod',suffix='1',settings=settings)
    atesa.makebatch(this_thread,settings=settings)
    assert my_file_cmp(settings.working_directory + '/fakethread_1_fwd.slurm', settings.working_directory + '/test_fwd.slurm')
    assert my_file_cmp(settings.working_directory + '/fakethread_1_bwd.slurm', settings.working_directory + '/test_bwd.slurm')
    os.remove(settings.working_directory + '/fakethread_1_fwd.slurm')
    os.remove(settings.working_directory + '/fakethread_1_bwd.slurm')
    os.remove(settings.logfile)


def test_makebatch_ca():
    settings = default_globals()
    settings.n_shots = 2
    settings.committor_analysis = [10,0.5,0.1,1,10,'test']
    this_thread = atesa.spawnthread('fakethread',thread_type='committor_analysis',suffix='1',settings=settings)
    atesa.makebatch(this_thread,settings=settings)
    assert my_file_cmp(settings.working_directory + '/fakethread_ca_0.slurm', settings.working_directory + '/test_ca_0.slurm')
    assert my_file_cmp(settings.working_directory + '/fakethread_ca_1.slurm', settings.working_directory + '/test_ca_1.slurm')
    os.remove(settings.working_directory + '/fakethread_ca_0.slurm')
    os.remove(settings.working_directory + '/fakethread_ca_1.slurm')
    os.remove(settings.logfile)


def test_checkcommit_as():
    settings = default_globals()
    this_thread = test_spawnthread_bare()
    settings.allthreads.append(this_thread)
    commit_flag = atesa.checkcommit(this_thread,'fwd','',settings)
    assert commit_flag == 'fwd'


def test_checkcommit_eps():
    settings = test_initialize_eps()
    this_thread = test_spawnthread_eps()
    settings.allthreads.append(this_thread)
    commit_flag = atesa.checkcommit(this_thread,'fwd','',settings)
    assert commit_flag == 'True'


def test_pickframe_on_thread():
    # Testing pickframe behavior when passed a thread, without fork
    settings = default_globals()
    this_thread = test_spawnthread_bare()
    settings.allthreads.append(this_thread)
    this_thread.last_valid = '1'
    new_file = atesa.pickframe(this_thread, 'fwd', frame=10, settings=settings)
    assert my_file_cmp(settings.working_directory + '/' + new_file, settings.working_directory + '/fakethread_1_fwd_frame_10.rst7')
    os.remove(settings.working_directory + '/' + new_file)
    os.remove(settings.logfile)


def test_pickframe_neg_n_adjust():
    # Testing pickframe behavior n_adjust is negative
    settings = default_globals()
    this_thread = test_spawnthread_bare()
    settings.allthreads.append(this_thread)
    settings.n_adjust = -10
    this_thread.last_valid = '1'
    new_file = atesa.pickframe(this_thread, 'fwd', settings=settings)
    assert my_file_cmp(settings.working_directory + '/' + new_file, settings.working_directory + '/fakethread_1_fwd_frame_10.rst7')
    os.remove(settings.working_directory + '/' + new_file)
    os.remove(settings.logfile)


def test_pickframe_on_string():
    # Testing pickframe behavior when passed a string, with fork
    settings = default_globals()
    this_thread = test_spawnthread_eps()
    settings.allthreads.append(this_thread)
    this_thread.last_valid = '1'
    new_file = atesa.pickframe(this_thread.name, 'fwd', forked_from=this_thread, frame=10, settings=settings)
    assert my_file_cmp(settings.working_directory + '/' + new_file, settings.working_directory + '/test.rst7_1_frame_10.rst7')
    os.remove(settings.working_directory + '/' + new_file)
    os.remove(settings.logfile)


def test_cleanthread_as():
    # Test cleanthread for an aimless shooting thread that has just been accepted
    settings = default_globals()
    this_thread = test_spawnthread_bare()
    settings.allthreads.append(this_thread)
    this_thread.last_valid = '1'    # suffix is also '1', so this means that the last move was accepted
    this_thread.commit1 = 'fwd'
    this_thread.commit2 = 'bwd'
    this_thread.accept_moves = 1
    cleanthread(this_thread, settings)
    assert my_file_cmp(settings.working_directory + '/status.txt', settings.working_directory + '/fakethread_status.txt')
    assert my_file_cmp(settings.working_directory + '/history/fakethread', settings.working_directory + '/fakethread_history')
    assert my_file_cmp(settings.working_directory + '/as.out', settings.working_directory + '/fakethread_as.out')
    os.remove(settings.working_directory + '/status.txt')
    shutil.rmtree(settings.working_directory + '/history')
    os.remove(settings.working_directory + '/as.out')
    os.remove(settings.working_directory + '/fakethread_2.rst7')
    os.remove(settings.logfile)


def test_cleanthread_eps():
    # Test cleanthread for an equilibrium path sampling thread that has just been accepted
    settings = test_initialize_eps()
    this_thread = test_spawnthread_eps()
    settings.allthreads.append(this_thread)
    this_thread.last_valid = '1'    # suffix is also '1', so this means that the last move was accepted
    this_thread.commit1 = 'True'
    this_thread.commit2 = 'True'
    this_thread.accept_moves = 1
    cleanthread(this_thread, settings)
    assert my_file_cmp(settings.working_directory + '/status.txt', settings.working_directory + '/eps.rst7_status.txt')
    assert my_file_cmp(settings.working_directory + '/history/eps.rst7', settings.working_directory + '/eps.rst7_history')
    # This next assert is done in this way because depending on random numbers the produced eps_results.out file may
    # only be a subset of the largest possible file.
    assert [(line in open(settings.working_directory + '/eps.rst7_eps_results.out').readlines()) for line in open(settings.working_directory + '/eps_results.out').readlines()]
    os.remove(settings.working_directory + '/status.txt')
    shutil.rmtree(settings.working_directory + '/history')
    os.remove(settings.working_directory + '/eps_results.out')
    try:
        os.remove(settings.working_directory + '/eps.rst7_2.rst7')
    except FileNotFoundError:
        pass    # happens when the init_fwd.rst file is chosen as the starting point for the next move
    os.remove(settings.logfile)


def test_cleanthread_dynamic_seed():
    # Test cleanthread with EPS, plus with eps_dynamic_seed
    settings = test_initialize_eps()
    settings.eps_dynamic_seed = True
    settings.empty_windows = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    this_thread = test_spawnthread_eps()
    settings.allthreads.append(this_thread)
    this_thread.last_valid = '1'    # suffix is also '1', so this means that the last move was accepted
    this_thread.commit1 = 'True'
    this_thread.commit2 = 'True'
    this_thread.accept_moves = 1
    this_thread.eps_fwd = 4         # to match actual lengths of trajectory files being used in tests
    this_thread.eps_bwd = 5
    cleanthread(this_thread, settings)
    assert my_file_cmp(settings.working_directory + '/status.txt', settings.working_directory + '/empty_windows_status.txt')
    assert my_file_cmp(settings.working_directory + '/history/eps.rst7', settings.working_directory + '/eps.rst7_history')
    assert not False in [(line[:45] in str(open(settings.working_directory + '/eps.rst7_eps_results.out').readlines())) for line in open(settings.working_directory + '/eps_results.out').readlines()]
    os.remove(settings.working_directory + '/status.txt')
    shutil.rmtree(settings.working_directory + '/history')
    os.remove(settings.working_directory + '/eps_results.out')
    try:
        os.remove(settings.working_directory + '/eps.rst7_2.rst7')
    except FileNotFoundError:
        pass    # happens when the init_fwd.rst file is chosen as the starting point for the next move
    os.remove(settings.working_directory + '/' + settings.logfile)
    os.remove(settings.working_directory + '/eps.rst7_1_1.rst7')
    os.remove(settings.working_directory + '/eps.rst7_1_2.rst7')


def test_candidatevalues_no_qdot():
    # Test for candidatevalues() with include_qdot = False
    settings = default_globals()
    cvs = atesa.candidatevalues('fakethread_1_init_fwd.rst', frame=-1, reduce=True, settings=settings)
    assert ['%.3f' % float(value) for value in cvs.split(' ')[:-1]] == ['0.753', '0.357']   # [:-1] removes trailing ''


def test_candidatevalues_with_qdot():
    # Test for candidatevalues() with include_qdot = True
    settings = default_globals()
    settings.rc_minmax = [[1, 0.6, 0.2, 0.3], [3, 2, 1.5, 1.75]]
    settings.include_qdot = True
    cvs = atesa.candidatevalues('fakethread_1_init_fwd.rst', frame=-1, reduce=True, settings=settings)
    assert ['%.3f' % float(value) for value in cvs.split(' ')[:-1]] == ['0.753', '0.357', '0.260', '0.966']   # [:-1] removes trailing ''


def test_candidatevalues_different_frame():
    # Test for candidatevalues() with a non-default parameter given for the frame argument (and reduce = False for good measure)
    settings = default_globals()
    cvs = atesa.candidatevalues('fakethread_1_fwd.nc', frame=2, reduce=False, settings=settings)
    assert ['%.3f' % float(value) for value in cvs.split(' ')[:-1]] == ['2.482', '1.244']   # [:-1] removes trailing ''


def test_candidatevalues_not_literal_ops():
    # Test for candidatevalues() using non-literal CV definitions
    settings = default_globals()
    settings.candidateops = [['@1','@2','@3'],['@4','@5','@6'],['','@7','@8'],['','','@9']]
    settings.literal_ops = False
    cvs = atesa.candidatevalues('fakethread_1_init_fwd.rst', frame=-1, reduce=False, settings=settings)
    assert ['%.3f' % float(value) for value in cvs.split(' ')[:-1]] == ['1.010', '91.137', '165.046']  # [:-1] removes trailing ''


def test_revvels():
    settings = default_globals()
    this_thread = test_spawnthread_bare()
    settings.allthreads.append(this_thread)
    atesa.revvels(this_thread)
    assert my_file_cmp(settings.working_directory + '/fakethread_1_init_bwd.rst',
                       settings.working_directory + '/fakethread_1_init_bwd_compare.rst')
    os.remove('fakethread_1_init_bwd.rst')
    os.remove('temp.rst')


def test_main_loop_as_no_groupfile():
    # Test of main_loop() behavior for an aimless shooting run without the groupfile setting. To facilitate this and
    # similar tests, spoofed output from subbatch() and interact() as well as spoofed output files in the working
    # directory are used.
    settings = default_globals()
    settings.DEBUGMODE = True
    this_thread = test_spawnthread_bare()
    settings.allthreads.append(this_thread)
    settings.itinerary = [this_thread]
    settings.max_moves = 1  # so the test will terminate after a single move
    atesa.main_loop(settings)
    assert my_file_cmp(settings.working_directory + '/fakethread_1_init.slurm',
                       settings.working_directory + '/fakethread_1_init_compare.slurm')
    assert my_file_cmp(settings.working_directory + '/fakethread_1_init.slurm',
                       settings.working_directory + '/fakethread_1_init_compare.slurm')
    assert my_file_cmp(settings.working_directory + '/restart.pkl',
                       settings.working_directory + '/restart_compare.pkl')
    os.remove(settings.working_directory + '/fakethread_1_init.slurm')
    os.remove(settings.working_directory + '/fakethread_1_init_bwd.rst')
    os.remove(settings.working_directory + '/fakethread_1_fwd.slurm')
    os.remove(settings.working_directory + '/fakethread_1_bwd.slurm')
    os.remove(settings.working_directory + '/as.out')
    shutil.rmtree(settings.working_directory + '/history')
    os.remove(settings.working_directory + '/restart.pkl')
    os.remove(settings.working_directory + '/status.txt')
    os.remove(settings.working_directory + '/' + settings.logfile)


# def test_handle_bootstrap_flag_true():
#     settings = default_globals()
#     settings.bootstrap_flag = True
#     assert atesa.handle_bootstrap(settings) == True


# def test_handle_bootstrap_make_jobs():
#     settings = default_globals()
#     settings.bootstrap_flag = False     # these next three settings are set in atesa.main()
#     settings.bootstrap_bookkeep = True
#     settings.bootstrap_jobids = []
#     settings.DEBUGMODE = True
#     shutil.copy(settings.working_directory + '/as_test.out', settings.working_directory + '/as.out')
#     atesa.handle_bootstrap(settings)
#     for i in range(settings.bootstrap_n + 1):
#         os.remove(settings.working_directory + '/bootstrap_2069_' + str(i) + '.slurm')
#         os.remove(settings.working_directory + '/bootstrap_2069_' + str(i) + '.in')
#     os.remove(settings.working_directory + '/' + settings.logfile)


# To make sure everything is running in the right place...
os.chdir(os.path.dirname(os.path.realpath(__file__)))
