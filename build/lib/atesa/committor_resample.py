# Standalone script to check commitment in every trajectory in the committor_analysis folder and find an estimate of pB
# for each group of trajectories. This is just for me, there should be no need to publish this, as the use case for this
# is just when rc_eval.committor_analysis() fucks up.

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

def main(input='committor_analysis'):
    commit_define_fwd = [['@7185','@7185'],['@7174','@7186'],[1.60,2.75],['lt','gt']]
    commit_define_bwd = [['@7185','@7185'],['@7174','@7186'],[2.75,2.00],['gt','lt']]

    def checkcommit(name,prmtop,directory=''):
        # Copied directly from atesa.py, with minor edits to make it standalone.

        committor_directory = ''
        if directory and not input:
            directory += '/'
            committor_directory = directory + 'committor_analysis/'
        elif input:
            committor_directory = input
            if not committor_directory[-1] == '/':
                committor_directory += '/'
            directory = committor_directory + '../'

        if not os.path.isfile(committor_directory + name):   # if the file doesn't exist yet, just do nothing
            return ''

        traj = pytraj.iterload(committor_directory + name, directory + prmtop, format='.nc')

        if not traj:                        # catches error if the trajectory file exists but has zero frames
            print('Don\'t worry about this internal error; it just means that ATESA is checking for commitment in a trajectory that doesn\'t have any frames yet, probably because the simulation has only just begun.')
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


    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    def update_progress(progress, message='Progress', eta=0):
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
        if eta:
            # eta is in seconds; convert into HH:MM:SS
            eta_h = str(math.floor(eta / 3600))
            eta_m = str(math.floor((eta % 3600) / 60))
            eta_s = str(math.floor((eta % 3600) % 60))
            if len(eta_m) == 1:
                eta_m = '0' + eta_m
            if len(eta_s) == 1:
                eta_s = '0' + eta_s
            eta_str = eta_h + ':' + eta_m + ':' + eta_s
            text = "\r" + message + ": [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block),
                                                              round(progress * 100, 2), status) + " ETA: " + eta_str
        else:
            text = "\r" + message + ": [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block),
                                                              round(progress * 100, 2), status)
        sys.stdout.write(text)
        sys.stdout.flush()


    working_directory = input#'/scratch/tburgin_flux/tburgin/as_working'

    try:
        os.chdir(working_directory)
    except FileNotFoundError:
        sys.exit('Error: could not find working directory: ' + working_directory)

    # try:
    #     os.chdir(input)
    # except FileNotFoundError:
    #     sys.exit('Error: could not find directory: ' + working_directory + '/' + input)

    trajectories = sorted(glob.glob('*_ca_*.nc'))
    # trajectories = []
    # for name in ['1.1_1.2_2.3_2.6.rst7_1_7.rst7', '1.4_1.3_2.2_2.7.rst7_1_13.rst7', '1.3_1.2_2.3_2.6.rst7_2_7.rst7', '1.4_1.3_2.2_2.7.rst7_2_9.rst7', '1.3_1.2_2.2_2.8.rst7_1_10.rst7', '1.3_1.2_2.4_2.7.rst7_1_25.rst7', '1.3_1.3_2.4_2.6.rst7_2_11.rst7', '1.3_1.2_2.3_2.6.rst7_2_5.rst7', '1.2_1.3_2.3_2.6.rst7_1_2.rst7', '1.3_1.2_2.3_2.6.rst7_1_28.rst7', '1.2_1.2_2.3_2.7.rst7_1', '1.2_1.2_2.3_2.7.rst7_2', '1.3_1.1_2.2_2.7.rst7_1_17.rst7', '1.3_1.4_2.3_2.6.rst7_1_10.rst7', '1.3_1.1_2.2_2.7.rst7_2_11.rst7', '1.3_1.3_2.3_2.6.rst7_2_13.rst7', '1.3_1.2_2.2_2.8.rst7_1_6.rst7', '1.2_1.2_2.3_2.5.rst7_2_17.rst7', '1.4_1.3_2.2_2.7.rst7_1_15.rst7', '1.4_1.2_2.3_2.6.rst7_1', '1.4_1.2_2.3_2.6.rst7_2', '1.1_1.2_2.3_2.6.rst7_1_19.rst7', '1.3_1.1_2.2_2.7.rst7_1_30.rst7', '1.2_1.2_2.4_2.7.rst7_1_10.rst7', '1.2_1.2_2.4_2.6.rst7_1_34.rst7', '1.3_1.3_2.2_2.5.rst7_2_9.rst7', '1.2_1.2_2.2_2.6.rst7_2_9.rst7', '1.4_1.2_2.2_2.7.rst7_1_23.rst7', '1.3_1.1_2.2_2.7.rst7_1_36.rst7', '1.3_1.3_2.4_2.7.rst7_2_16.rst7', '1.3_1.2_2.4_2.7.rst7_1_22.rst7', '1.3_1.3_2.1_2.6.rst7_2_8.rst7']:
    #     trajectories += glob.glob(name + '_ca_*.nc')

    with open('committor_analysis.out', 'w') as f:
        f.close()

    last_basename = ''
    this_committor = []
    count = 0
    speed_values = []
    count_to = len(trajectories)
    for trajectory in trajectories:
        t = time.time()
        basename = trajectory[:-8]  # name excluding '_ca_#.nc' where # is one digit (only works because mine are 0-9)
        commit = checkcommit(trajectory,'ts_guess.prmtop',working_directory)
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
                if fwd_count + bwd_count >= 2:
                    try:
                        pb = fwd_count/(fwd_count+bwd_count)
                        open('committor_analysis.out', 'a').write(basename + ': ' + str(pb) + '\n')
                        open('committor_analysis.out', 'a').close()
                    except ZeroDivisionError:
                        pass
            # start new basename
            last_basename = basename
            this_committor = [commit]
        else: #if basename == last_basename
            this_committor.append(commit)
        speed_values.append(time.time() - t)
        speed = numpy.mean(speed_values)
        count += 1
        eta = (count_to - count) * speed
        update_progress(count / len(trajectories), message='Checking commitment of committor analysis trajectories',eta=eta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform resampling of committor analysis data.')
    parser.add_argument('-i', metavar='input_directory', type=str, nargs=1, default='committor_analysis',
                        help='Committor analysis directory name. Default=\'committor_analysis\'')
    arguments = vars(parser.parse_args())  # Retrieves arguments as a dictionary object

    main(arguments.get('i')[0])
