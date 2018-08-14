'''
    File name: ilmax.py
    Author: Justin Huber
    Date created: 7/3/2018
    Date last modified: x/x/xxxx
    Python Version: 3.x
'''

from __future__ import print_function
import multiprocessing
import math
import numpy as np
import pandas as pd
from statsmodels.base.model import GenericLikelihoodModel
import argparse
import time
import queue
import sys
# from aimless_shooting import update_progress
from sympy import Matrix
from sympy import lambdify
import warnings


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress, message='Progress'):
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
# import cProfile       # performance profiling


def parseOutcomeFile(outcome_file):
    '''
    read the outcome file into variables to be passed to ILMaxModel
    
    :param outcome_file: 
    :return: 
    '''

    num_A_outcomes, num_B_outcomes = outcome_file.readline().split()
    num_A_outcomes = int(num_A_outcomes)
    num_B_outcomes = int(num_B_outcomes)

    num_q_params, num_qdot_params = outcome_file.readline().split()
    num_q_params= int(num_q_params)
    num_qdot_params= int(num_qdot_params)

    # remove space
    outcome_file.readline()

    A_lines = []
    B_lines = []

    column_names = []

    for i in range(num_q_params + num_qdot_params):
        column_names.append("var" + str(i + 1))

    for i in range(num_A_outcomes):
        A_lines.append(outcome_file.readline())

    for i in range(num_B_outcomes):
        B_lines.append(outcome_file.readline())

    return num_A_outcomes, num_B_outcomes,\
           num_q_params, num_qdot_params,\
           column_names,\
           A_lines, B_lines

def parseABData(A_data, B_data):
    '''
    reduce variables
    
    :param A_data: 
    :param B_data: 
    :return: 
    '''

    qmax = []
    qmin = []

    for i in range(len(A_data)):
        qmax.append(A_data[i][0])
        qmin.append(A_data[i][0])

        for j in range(len(A_data[i])):     # update qmax and qmin using A_data
            if qmax[i] < A_data[i][j]:
                qmax[i] = A_data[i][j]

            if qmin[i] > A_data[i][j]:
                qmin[i] = A_data[i][j]

        for j in range(len(B_data[i])):     # update qmax and qmin using B_data
            if qmax[i] < B_data[i][j]:
                qmax[i] = B_data[i][j]

            if qmin[i] > B_data[i][j]:
                qmin[i] = B_data[i][j]

    for i in range(len(A_data)):            # REDUCED VARIABLES: Z = (Q-Qmin)/(Qmax-Qmin)
        for j in range(len(A_data[i])):
            A_data[i][j] = (A_data[i][j] - qmin[i]) / (qmax[i] - qmin[i])

        for j in range(len(B_data[i])):
            B_data[i][j] = (B_data[i][j] - qmin[i]) / (qmax[i] - qmin[i])

def parseData(lines, num_params):
    '''
    converts strings to lists of floats
    
    :param lines: 
    :param num_params: 
    :return: 
    '''
    data = [[] for x in range(num_params)]

    for x in lines:
        x = x.strip()           # get rid of \n at end of each line
        x = x.split(' ')        # split var's into an array
        x = np.asarray(x)       # convert to np array
        x = x.astype(np.float)  # convert array to floats

        for i in range(len(x)):
            data[i].append(x[i])    # add data to array

    return data

class ILMaxModel:
    def __init__(self, k_best=None, outcome_filename="outcomes.txt", method="Powell", **kwds):
        start_time = time.time()
        '''
        Class to create a model using ILMax and BIC for culling
        
        :param outcome_filename: input filename containing the outcome data
        :param kwds: additional keywords to be passed to GenericLikelihoodModel superclass
        '''

        outcome_file = open(outcome_filename, 'r')

        num_A_outcomes, num_B_outcomes, \
        num_q_params, num_qdot_params, \
        column_names,\
        A_lines, B_lines = parseOutcomeFile(outcome_file)

        outcome_file.close()

        num_params = num_q_params + num_qdot_params

        A_data = parseData(A_lines, num_params)
        B_data = parseData(B_lines, num_params)

        parseABData(A_data, B_data)

        self.A_data = A_data
        self.B_data = B_data

        ilmax_args = queue.Queue()      # queue of arguments to be passed to threadedILMax when running parallel

        if k_best is None:              # if k_best isn't defined, default to entire model
            k_best = num_q_params

        self.count = 0
        self.num_comb = 0
        for combination in combinations(num_q_params, k_best):
            self.num_comb += 1

        models = []         # list of model data

        num_cores = multiprocessing.cpu_count()
        # print("Number of cpu cores: ", num_cores) # TODO add to verbose option
        if (num_cores > 2):             # run in parallel if more than 2 cores
            runInParallel = True
        else:                           # run in series if 2 cores or less
            runInParallel = False

        # pr = cProfile.Profile()
        # pr.enable()

        for combination in combinations(num_q_params, k_best):                      # TODO add support for qdot terms
            exog_data = [ [] for x in range(num_A_outcomes + num_B_outcomes) ]      # list of observation OP data

            for i in range(num_A_outcomes):                     # append A outcome observations
                for j in combination:
                    exog_data[i].append(A_data[j][i])

            for i in range(num_B_outcomes):                     # append B outcome observations
                for j in combination:
                    exog_data[num_A_outcomes + i].append(B_data[j][i])

            new_column_names = []                               # list of parameter names for current model

            for j in combination:                               # create list of parameter names for current iteration
                new_column_names.append(column_names[j])

            exog = pd.DataFrame(exog_data, columns=new_column_names)
            endog = pd.DataFrame(np.zeros(exog.shape))          # matrix of zeroes with same shape as exog

            ilmax_args.put((combination,
                                exog, endog,
                                num_A_outcomes, num_B_outcomes,
                                len(new_column_names), num_qdot_params,
                                method, multiprocessing.Queue()))

            if not runInParallel:            # run in series
                ilmax_fit = self.threadedILMax(combination,
                                            exog, endog,
                                            num_A_outcomes, num_B_outcomes,
                                            len(new_column_names), num_qdot_params,
                                            method, multiprocessing.Queue())

                models.append((combination,ilmax_fit.bic,ilmax_fit.llf,ilmax_fit.mle_retvals['converged'],ilmax_fit.params))

        if runInParallel:                   # run in parallel
            results_queue = multiprocessing.Queue()

            jobs = []
            while not ilmax_args.empty():
                arg = ilmax_args.get()
                p = multiprocessing.Process(target=self.threadedILMax, args=(arg[0],
                                                                        arg[1], arg[2],
                                                                        arg[3], arg[4],
                                                                        arg[5], arg[6],
                                                                        method, results_queue))
                jobs.append(p)
                p.start()

                update_progress(self.count / self.num_comb, str(self.count) +
                                ' out of ' + str(self.num_comb) + ' combinations fitted')

                self.count += 1

            for proc in jobs:
                proc.join()

            while not results_queue.empty():
                models.append(results_queue.get())

        models = sorted(models, key=lambda model: model[1])

        # pr.disable()
        # pr.print_stats(sort='tottime')        # after program ends

        file = open(outcome_filename.split('.')[0] + '-models.txt', 'w')
        file.write('combination BIC loglikelihood converged coefficients\n')
        for tuple in models:
            for item in tuple:
                file.write(str(item) + ' ')
            file.write('\n')

        file.write("\nExecuted in " + str(time.time() - start_time) + "s\n")

    def threadedILMax(self, combination,
                      exog, endog,  # TODO currently has no support for passing **kwds to ILMax
                      num_A_outcomes, num_B_outcomes,
                      num_q_params, num_qdot_params,
                      method,
                      results_queue, **kwds):
        '''

        :param exog: 
        :param endog: 
        :param num_A_outcomes: 
        :param num_B_outcomes: 
        :param num_q_params: 
        :param num_qdot_params: 
        :param results_queue: 
        :param kwds: 
        :return: ilmax_fit:
        '''
        ilmax = ILMax(exog=exog, endog=endog,  # inertial likelihood maximization for current model
                      num_A_outcomes=num_A_outcomes,
                      num_B_outcomes=num_B_outcomes,
                      num_q_params=num_q_params,
                      num_qdot_params=num_qdot_params,
                      **kwds)

        start_params = [0 for null in range(num_q_params + num_qdot_params + 1)]
        ilmax_fit = ilmax.fit(method=method, start_params=start_params) # TODO: add disp for verbose

        # print('Current combination: ', combination)  # current combination of k_best OP's # TODO verbose option?
        # print('BIC: ', ilmax_fit.bic)
        # print('CONVERGED: ', ilmax_fit.mle_retvals['converged'])  # True or False
        # print('MODEL COEFFICIENTS: ', ilmax_fit.params)  # Coefficients of the model
        # print(ilmax_fit.summary())  # Detailed output

        results_queue.put(
            (combination, ilmax_fit.bic, ilmax_fit.llf, ilmax_fit.mle_retvals['converged'], ilmax_fit.params))
        return ilmax_fit


def thetaB(exog_i, params):
    '''
    Eqn 20 of PAPER_LINK
    :param exog_i: order parameters at shooting point i
    :param params: coefficients
    :return: thetaB
    '''
    erf_arg = -params[0]

    for i in range(len(exog_i)):
        erf_arg += exog_i[i] * params[i + 1]

    thetaB = (1 + math.erf(erf_arg))/2

    return thetaB

class ILMax(GenericLikelihoodModel):
    def __init__(self, endog, exog,
                 num_A_outcomes, num_B_outcomes,
                 num_q_params, num_qdot_params,
                 **kwds):
        '''
        Performs inertial likelihood maximization on a set of data
        
        :param endog: 
        :param exog: 
        :param num_A_outcomes: 
        :param num_B_outcomes: 
        :param num_q_params: 
        :param num_qdot_params: 
        :param kwds: 
        '''

        self.num_A_outcomes = num_A_outcomes
        self.num_B_outcomes = num_B_outcomes
        self.num_q_params = num_q_params
        self.num_qdot_params = num_qdot_params
        super(ILMax, self).__init__(endog, exog, **kwds)

        # TODO add a check that all columns are linearly independent
        # TODO and remove columns which aren't independent
        rank = np.linalg.matrix_rank(self.exog)
        expected_rank = num_q_params + num_qdot_params

        assert (rank == expected_rank)          # TODO check for linear independent data and implement functionality for linearly dependent data

        # matrix = Matrix(self.exog)
        # rref_tuple = matrix.rref()
        #
        # rref = rref_tuple[0].tolist()

    def loglike(self, params):
        '''
        Eqn 21 of PAPER_LINK
        :param params: 
        :return: 
        '''
        exog = self.exog

        sum = 0

        # iterate through A outcomes
        for i in range(self.num_A_outcomes):
            sum += np.log(1 - thetaB(exog[i], params))

        # iterate through B outcomes
        for i in range(self.num_B_outcomes):
            sum += np.log(thetaB(exog[self.num_A_outcomes + i], params))

        return sum

    def fit(self, start_params=None, method='Powell', maxiter=10000, maxfun=5000, disp=False, **kwds):
        maxiter *= self.num_q_params            # Currently maxiter and maxfun are functions of the parameter space
        maxfun *= self.num_q_params             # Somewhat arbitrary / somewhat empirically determined
        if 'Powell' in method:
            maxfun = None

        # we have one additional parameter and we need to add it for summary
        self.exog_names.insert(0, 'alpha0')
        if start_params == None:
            # Reasonable starting values
            start_params = np.ones(self.exog.shape[1] + 1)

        for i in range(len(start_params)):          # divide by the number of observations to ensure the sum is < 1
            start_params[i] /= (self.exog.shape[1] + 1)       # prevents potential log(0) error in loglike()

        if maxfun is None:
            return super(ILMax, self).fit(start_params=start_params, method="minimize",
                                          maxiter=maxiter, min_method=method,
                                          disp=disp, skip_hessian=True,
                                          **kwds)  # TODO: add skip_hessian functionality
        else:
            return super(ILMax, self).fit(start_params=start_params, method="minimize",
                                         maxiter=maxiter, maxfun=maxfun, min_method=method,
                                         disp=disp, skip_hessian=True,
                                         **kwds) # TODO: add skip_hessian functionality


def generate_random_data(num_params, num_obs):
    file = open('invertibility_testing/randdata.txt', 'w')

    file.write(str(int(np.floor(num_obs/2))) + ' ' + str(int(np.ceil(num_obs/2))) + '\n')
    file.write(str(num_params) + ' 0\n\n')

    for i in range(num_obs):
        for j in range(num_params):
            file.write(str(np.random.uniform()) + ' ')
        file.write('\n')

def get_pos_to_change(comb, n):
    """
    Finds the rightmost position in the comb list such that its value can
    can be increased. Returns -1 if there's no such position.
    """
    pos_to_change = len(comb) - 1
    max_possible_value = n - 1
    while pos_to_change >= 0 and comb[pos_to_change] == max_possible_value:
        pos_to_change -= 1
        max_possible_value -= 1
    return pos_to_change

def inc_value_at_pos(comb, pos):
    """
    Increments the value at the given position and generates the 
    lexicographically smallest suffix after this position.
    """
    new_comb = comb[:]
    new_comb[pos] += 1
    for idx in range(pos + 1, len(comb)):
       new_comb[idx] = new_comb[idx - 1] + 1
    return new_comb

def get_next_combination(comb, n):
    """
    Returns the lexicographically next combination or None if it doesn't
    exist.
    """
    pos_to_change = get_pos_to_change(comb, n)
    if pos_to_change < 0:
        return None
    return inc_value_at_pos(comb, pos_to_change)

def combinations(n, k):
    """
    Generates all n choose k combinations of the n natural numbers
    """
    comb = [i for i in range(k)]
    while comb is not None:
        yield comb
        comb = get_next_combination(comb, n)

def main():
    # generate_random_data(10, 2069) # TODO potentially remove
    start_time = time.time()

    # Parse arguments from command line using argparse
    # TODO replace PLACEHOLDER
    parser = argparse.ArgumentParser(description='PLACEHOLDER')
    parser.add_argument('-k', '--k_best', type=int, default=None,
                        help='set the desired model OP dimensionality (expects int)\n'
                             'NOTE: not setting this will skip the model reduction step,\n')
    parser.add_argument('-m', '--method', type=str, default="Powell",
                        help='set desired optimization method')             # TODO add list of method choices
    parser.add_argument('-i', '--inputfile', type=str,
                        help='input filename (required)', required=True)    # TODO CHANGE TO NOT BE REQUIRED?
    args = parser.parse_args()

    ILMaxModel(k_best=args.k_best, outcome_filename=args.inputfile, method=args.method)

    print("Executed in " + str(time.time() - start_time) + "s")         # Program's execution time

if __name__ == "__main__":
    main()          # execute only if run as a script
