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
import matplotlib.pyplot as plt
from statsmodels.base.model import GenericLikelihoodModel
import argparse
import time
from sympy import Matrix
from sympy import lambdify
import warnings


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

def parseData(lines, data):
    '''
    converts strings to lists of floats
    
    :param lines: 
    :param data: 
    :return: 
    '''
    for x in lines:
        # get rid of \n at end of each line
        x = x.strip()
        # split var's into an array
        x = x.split(' ')
        # convert to np array
        x = np.asarray(x)
        # convert array to floats
        x = x.astype(np.float)

        # add data to array
        for i in range(len(x)):
            data[i].append(x[i])

    # reduce variables
    qmax = []
    qmin = []

    for i in range(len(data)):
        qmax.append(data[i][0])
        qmin.append(data[i][0])

        for j in range(len(data[i])):
            if qmax[i] < data[i][j]:
                qmax[i] = data[i][j]

            if qmin[i] > data[i][j]:
                qmin[i] = data[i][j]

    # REDUCED VARIABLES: Z = (Q-Qmin)/(Qmax-Qmin)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - qmin[i]) / (qmax[i] - qmin[i])

    return data


def threadedILMax(exog, endog,                      # TODO currently has no support for passing **kwds to ILMax
                  num_A_outcomes, num_B_outcomes,
                  num_q_params, num_qdot_params,
                  method,
                  param, results_queue, **kwds):
    '''
    
    :param exog: 
    :param endog: 
    :param num_A_outcomes: 
    :param num_B_outcomes: 
    :param num_q_params: 
    :param num_qdot_params: 
    :param param: 
    :param results_queue: 
    :param kwds: 
    :return: 
    '''

    ilmax = ILMax(exog=exog, endog=endog,           # inertial likelihood maximization for current model
                  num_A_outcomes=num_A_outcomes,
                  num_B_outcomes=num_B_outcomes,
                  num_q_params=num_q_params,
                  num_qdot_params=num_qdot_params,
                  **kwds)

    start_params = [0 for null in range(num_q_params+num_qdot_params+1)]
    ilmax_fit = ilmax.fit(method=method, start_params=start_params)
    print('\nBIC: ', ilmax_fit.bic)
    print('\nCONVERGED: ', ilmax_fit.mle_retvals['converged']) # True or False
    print('\nMODEL COEFFICIENTS: ', ilmax_fit.params)                   # Coefficients of the model
    # print(ilmax_fit.summary())                      # Detailed output
    # print(ilmax_fit.cov_params(), '\n')           # DEPRACATED

    # BIC used to compare models
    BIC = ilmax_fit.bic

    results_queue.put((param, BIC))
    return ilmax_fit

class ILMaxModel:
    def __init__(self, k_best=None, outcome_filename="outcomes.txt", method="powell", **kwds):
        '''
        Class to create a model using ILMax and BIC/AIC for culling
        
        :param outcome_filename: input filename containing the outcome data
        :param kwds: additional keywords to be passed to GenericLikelihoodModel superclass
        '''

        self.candidate_params = []      # candidate q parameters that will iteratively
                                        # be removed to test if model can be reduced
                                        # a model can be reduced if its BIC is lower than the best model's BIC

        outcome_file = open(outcome_filename, 'r')

        num_A_outcomes, num_B_outcomes, \
        num_q_params, num_qdot_params, \
        column_names,\
        A_lines, B_lines = parseOutcomeFile(outcome_file)

        outcome_file.close()

        num_params = num_q_params + num_qdot_params

        A_data = [ [] for x in range(num_params) ]
        B_data = [ [] for x in range(num_params) ]

        A_data = parseData(A_lines, A_data)
        B_data = parseData(B_lines, B_data)

        self.A_data = A_data
        self.B_data = B_data

        self.candidate_params = np.arange(-1, num_q_params)     # [ -1, 0, 1, ..., num_q_params - 1 ]

        ilmax_args = []                                         # list of arguments to be passed to threadedILMax

        if k_best is None:              # if k_best isn't defined, default to entire model
            k_best = num_q_params

        models = []

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

            ilmax_args.append((exog, endog,                     # threadILMAX arguments (in form of a tuple)
                              num_A_outcomes, num_B_outcomes,
                              len(new_column_names), num_qdot_params,
                              self.candidate_params[0]))

            print('Current combination: ', combination)              # current combination of k_best OP's
            ilmax_fit = threadedILMax(exog, endog,
                          num_A_outcomes, num_B_outcomes,
                          len(new_column_names), num_qdot_params,
                          method,
                          self.candidate_params[0], multiprocessing.Queue())

            models.append((combination,ilmax_fit.bic,ilmax_fit.mle_retvals['converged'],ilmax_fit.params))

        models = sorted(models, key=lambda model: model[1])

        file = open(outcome_filename.split('.')[0] + '-models.txt', 'w')
        file.write('combination BIC converged coefficients\n')
        for tuple in models:
            for item in tuple:
                file.write(str(item) + ' ')
            file.write('\n')
        # while len(self.candidate_params) > 0:                   # iterate over all candidate parameters
        #     exog_data = [ [] for x in range(num_A_outcomes + num_B_outcomes) ]      # list of observation OP data
        #
        #     for i in range(num_A_outcomes):                     # append A outcome observations
        #         for j in range(num_params):
        #             if j != self.candidate_params[0]:
        #                 exog_data[i].append(A_data[j][i])
        #
        #     for i in range(num_B_outcomes):                     # append B outcome observations
        #         for j in range(num_params):
        #             if j != self.candidate_params[0]:
        #                 exog_data[num_A_outcomes + i].append(B_data[j][i])
        #
        #     new_column_names = []                               # list of parameter names for current model
        #
        #     for i in range(len(column_names)):                  # create list of parameter names for current iteration
        #         if i != self.candidate_params[0]:
        #             new_column_names.append(column_names[i])
        #
        #     exog = pd.DataFrame(exog_data, columns=new_column_names)
        #     endog = pd.DataFrame(np.zeros(exog.shape))          # matrix of zeroes with same shape as exog
        #
        #     ilmax_args.append((exog, endog,                     # threadILMAX arguments (in form of a tuple)
        #                       num_A_outcomes, num_B_outcomes,
        #                       len(new_column_names), num_qdot_params,
        #                       self.candidate_params[0]))
        #
        #     if k_best is None:      # if no k_best provided, default to entire OP set
        #         threadedILMax(exog, endog,
        #                       num_A_outcomes, num_B_outcomes,
        #                       len(new_column_names), num_qdot_params,
        #                       method,
        #                       self.candidate_params[0], multiprocessing.Queue())
        #         return              # return early since there's no need to compare BIC's for the entire OP set
        #
        #     self.candidate_params = np.delete(self.candidate_params, 0)     # move on to next candidate parameter

        # results_list = []
        # results_queue = multiprocessing.Queue()
        #
        # print("Number of cpu cores: ", multiprocessing.cpu_count())         # running in parallel
        # jobs = []
        # for arg in ilmax_args:
        #     p = multiprocessing.Process(target=threadedILMax, args=(arg[0], arg[1],
        #                                                             arg[2], arg[3],
        #                                                             arg[4], arg[5],
        #                                                             method,
        #                                                             arg[6], results_queue))
        #     jobs.append(p)
        #     p.start()
        #
        # for proc in jobs:
        #     proc.join()
        #
        # while not results_queue.empty():
        #     results_list.append(results_queue.get())
        #
        # # for arg in ilmax_args:                                            # running in series
        # #     results_list.append(threadedILMax(arg[0], arg[1],
        # #                                         arg[2], arg[3],
        # #                                         arg[4], arg[5],
        # #                                         method,
        # #                                         arg[6], results_queue))
        #
        # results_list.sort()
        #
        # dBIC = []
        #
        # i = 1
        # while i < len(results_list):                # create list of (refBIC - BIC, param_index) tuples
        #     dbic = results_list[0][1] - results_list[i][1]
        #     dBIC.append((dbic, i - 1))
        #
        #     i += 1
        #
        # dBIC.sort()             # sort (sorts based on first tuple element in ascending order by default)
        #
        # k_best_params = []
        # k_best_param_names = []
        # k_best_exog_data = [[] for x in range(num_A_outcomes + num_B_outcomes)]
        #
        # for k in range(k_best):                     # create model from k best parameters
        #     k_best_params.append(dBIC[k][1])
        #     k_best_param_names.append(column_names[dBIC[k][1]])
        #
        # for i in range(num_A_outcomes):             # append A outcome observations
        #     for j in range(len(k_best_params)):
        #         k_best_exog_data[i].append(A_data[k_best_params[j]][i])
        #
        # for i in range(num_B_outcomes):             # append B outcome observations
        #     for j in range(len(k_best_params)):
        #         k_best_exog_data[num_A_outcomes + i].append(B_data[k_best_params[j]][i])
        #
        # k_best_exog = pd.DataFrame(k_best_exog_data, columns=k_best_param_names)
        # k_best_endog = pd.DataFrame(np.zeros(k_best_exog.shape))
        #
        # k_best_ilmax = ILMax(exog=k_best_exog, endog=k_best_endog,
        #               num_A_outcomes=num_A_outcomes,
        #               num_B_outcomes=num_B_outcomes,
        #               num_q_params=(len(k_best_param_names)),
        #               num_qdot_params=num_qdot_params,
        #               **kwds)
        #
        # k_best_ilmax_fit = k_best_ilmax.fit(method=method)
        #
        # print('\n', k_best_ilmax_fit.params)
        # print(k_best_ilmax_fit.summary())
        # # print(k_best_ilmax_fit.cov_params(), '\n')            # TODO DEPRACATED

    # # TODO outdated
    # def get_outcome_thetaB(self, outcomes):
    #     '''
    #     reaction coordinate from thetaB for outcomes
    #
    #     :param outcomes:
    #     :return:
    #     '''
    #     outcome_thetaB = []
    #
    #     outcome_data = [[] for x in range(len(outcomes[0]))]
    #
    #     # append outcome observations
    #     for i in range(len(outcomes[0])):
    #         for j in range(len(outcomes)):
    #             # if the current parameter is significant
    #             if j not in self.excluded_params:
    #                 outcome_data[i].append(outcomes[j][i])
    #
    #     for i in range(len(outcome_data)):
    #         outcome_thetaB.append(thetaB(outcome_data[i], self.best_model_fit.params))
    #
    #     return outcome_thetaB

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

    def fit(self, start_params=None, method='powell', maxiter=10000, maxfun=5000, **kwds):
        maxiter *= self.num_q_params            # Currently maxiter and maxfun are functions of the parameter space
        maxfun *= self.num_q_params             # Somewhat arbitrary / somewhat empirically determined

        # we have one additional parameter and we need to add it for summary
        self.exog_names.insert(0, 'alpha0')
        if start_params == None:
            # Reasonable starting values
            start_params = np.ones(self.exog.shape[1] + 1)

        for i in range(len(start_params)):          # divide by the number of observations to ensure the sum is < 1
            start_params[i] /= (self.exog.shape[1] + 1)       # prevents potential log(0) error in loglike()

        return super(ILMax, self).fit(start_params=start_params, method="minimize",
                                     maxiter=maxiter, maxfun=maxfun,min_method=method,
                                     **kwds)

def generate_random_data(num_params, num_obs):
    file = open('invertibility_testing/randdata.txt', 'w')

    file.write(str(int(np.floor(num_obs/2))) + ' ' + str(int(np.ceil(num_obs/2))) + '\n')
    file.write(str(num_params) + ' 0\n\n')

    for i in range(num_obs):
        for j in range(num_params):
            file.write(str(np.random.uniform()) + ' ')
        file.write('\n')

# # TODO outdated
# def plot_model_thetaB_sigmoid(outcome_filename, bin_width=0.1):
#     '''
#     plots the thetaB's of simplified model
#
#     :param outcome_filename:
#     :param bin_width:
#     :return:
#     '''
#     ilmax = ILMaxModel(outcome_filename)
#
#     a_thetaB = ilmax.get_outcome_thetaB(ilmax.A_data)
#     b_thetaB = ilmax.get_outcome_thetaB(ilmax.B_data)
#
#     a_thetaB = sorted(a_thetaB)
#     b_thetaB = sorted(b_thetaB)
#
#     plot_thetaB_sigmoid(a_thetaB, b_thetaB, bin_width=bin_width, title=outcome_filename)
#
# # TODO outdated
# #excluding sigmoid fit until decided it's important
# # # sigmoid shape for domain and range of [0, 1]
# # # beta can be varied to change the slope of the sigmoid
# # def sigmoid(x, beta=3):
# #     for i in range(len(x)):
# #         x[i] = 1 / (1 + pow((x[i]/((1-x[i]))),-beta))
# #
# #     return x
#
# # TODO outdated
# # sigmoid plotting of thetaB data with a specified bin size
# def plot_thetaB_sigmoid(a_thetaB, b_thetaB, bin_width=0.1, title="UNTITLED"):
#     # convert list of floats to a list of tuples of the form (float, outcome)
#     for i in range(len(a_thetaB)):
#         a_thetaB[i] = (a_thetaB[i], 'A')
#
#     for i in range(len(b_thetaB)):
#         b_thetaB[i] = (b_thetaB[i], 'B')
#
#     # combine the lists
#     thetaB = a_thetaB + b_thetaB
#     # sort (sorts based on first tuple element in ascending order by default)
#     thetaB.sort()
#
#     num_going_to_B = 0
#
#     bins = np.zeros(int(1/bin_width))
#
#     # start at bin with smallest thetaB val
#     curr_bin = int(thetaB[0][0] / bin_width)
#     bin_tot = 0
#
#     for i in range(len(thetaB)):
#         bin_tot += 1
#         if thetaB[i][0] > (curr_bin + 1) * bin_width:
#             bins[curr_bin] = num_going_to_B / bin_tot
#             num_going_to_B = 0
#             bin_tot = 1
#             curr_bin += 1
#
#         if thetaB[i][1] == 'B':
#             num_going_to_B += 1
#
#     bins[curr_bin] = num_going_to_B / bin_tot
#
#     x_pos = np.arange(0, 1, bin_width)
#
#     plt.bar(x_pos, bins, align='edge', width=bin_width, alpha=0.5)
#     plt.xticks(x_pos)
#     plt.xlabel("thetaB")
#     plt.ylabel('Fraction going to B')
#     plt.title(title)
#
#     #excluding sigmoid fit until decided it's important
#     # # plotting comparison sigmoid
#     # # linespace generate an array from start and stop value
#     # x = np.linspace(0, 1, 100)
#     # y = np.linspace(0, 1, 100)
#     # y = sigmoid(y)
#     #
#     # # prepare the plot, associate the color r(ed) or b(lue) and the label
#     # plt.plot(x, y, 'r')
#
#     plt.show()

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
    generate_random_data(10, 2069)
    start_time = time.time()

    # Parse arguments from command line using argparse
    # TODO replace PLACEHOLDER
    parser = argparse.ArgumentParser(description='PLACEHOLDER')
    parser.add_argument('-k', '--k_best', type=int, default=None,
                        help='set the desired model OP dimensionality (expects int)\n'
                             'NOTE: not setting this will skip the model reduction step,\n'
                             'greatly reducing computation time for systems with many OPs')
    parser.add_argument('-m', '--method', type=str, default="lbfgs",
                        help='set desired optimization method')
    parser.add_argument('-i', '--inputfile', type=str,
                        help='input filename (required)', required=True)  # TODO CHANGE TO NOT BE REQUIRED?
    args = parser.parse_args()

    ILMaxModel(k_best=args.k_best, outcome_filename=args.inputfile, method=args.method)

    print("Executed in " + str(time.time() - start_time) + "s")         # Program's execution time

if __name__ == "__main__":
    main()          # execute only if run as a script
