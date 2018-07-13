'''
    File name: ilmax.py
    Author: Justin Huber
    Date created: 7/3/2018
    Date last modified: x/x/xxxx
    Python Version: 3.x
'''

from __future__ import print_function
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.base.model import GenericLikelihoodModel
import warnings


# ILMaxModel
# read the outcome file into variables to be passed to ILMaxModel
def parseOutcomeFile(outcome_file):
    num_A_outcomes, num_B_outcomes = outcome_file.readline().split()
    num_A_outcomes = int(num_A_outcomes)
    num_B_outcomes = int(num_B_outcomes)

    num_q_params, num_qdot_params = outcome_file.readline().split()
    num_q_params= int(num_q_params)
    num_qdot_params= int(num_qdot_params)

    # remove space
    outcome_file.readline()
    column_names = outcome_file.readline().split()

    A_lines = []
    B_lines = []

    for i in range(num_A_outcomes):
        A_lines.append(outcome_file.readline())

    for i in range(num_B_outcomes):
        B_lines.append(outcome_file.readline())

    return num_A_outcomes, num_B_outcomes,\
           num_q_params, num_qdot_params,\
           column_names,\
           A_lines, B_lines

# converts strings to lists of floats
def parseData(lines, data):
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

# Class to create a model using ILMax and BIC/AIC for culling
class ILMaxModel:
    def __init__(self, outcome_filename="outcomes.txt", **kwds):
        '''
        
        :param outcome_filename: input filename containing the outcome data
        :param kwds: additional keywords to be passed to GenericLikelihoodModel superclass
        '''
        # candidate q parameters that will iteratively removed to test if model can be reduced
        # a model can be reduced if its BIC is lower than the best model's BIC
        self.candidate_params = []
        # parameters than can be excluded from the model
        self.excluded_params = []

        # initialize to infinity to ensure first round keeps all parameters
        self.old_BIC = np.Infinity
        self.old_AIC = np.Infinity

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

        # an array of the indices to the parameters
        # starts at -1 to initialize the first model with all parameters
        self.candidate_params = np.arange(-1, num_q_params)

        # iterate over candidate parameters to find simplified model
        while len(self.candidate_params) > 0:
            # a list of the observations to be used to create the likelihood maximization
            exog_data = [ [] for x in range(num_A_outcomes + num_B_outcomes) ]

            # append A outcome observations
            for i in range(num_A_outcomes):
                for j in range(num_params):
                    # if the current parameter is significant
                    # and it isn't the current candidate for comparison
                    if j not in self.excluded_params and j != self.candidate_params[0]:
                        exog_data[i].append(A_data[j][i])

            # append B outcome observations
            for i in range(num_B_outcomes):
                for j in range(num_params):
                    # if the current parameter is significant
                    # and it isn't the current candidate for comparison
                    if j not in self.excluded_params and j != self.candidate_params[0]:
                        exog_data[num_A_outcomes + i].append(B_data[j][i])

            # list of parameter names for current model
            new_column_names = []

            for i in range(len(column_names)):
                if i not in self.excluded_params and i != self.candidate_params[0]:
                    new_column_names.append(column_names[i])

            exog = pd.DataFrame(exog_data, columns=new_column_names)
            # initialized to matrix of zeroes with same shape as exog
            # avoids errors with GeneralLikelihoodModel
            endog = pd.DataFrame(np.zeros(exog.shape))

            # inertial likelihood maximization for current model
            ilmax = ILMax(exog=exog, endog=endog,
                               num_A_outcomes=num_A_outcomes,
                               num_B_outcomes=num_B_outcomes,
                               num_q_params=(num_q_params - len(self.excluded_params)),
                               num_qdot_params=num_qdot_params,
                               **kwds)

            ilmax_fit = ilmax.fit(method='bfgs')

            # output for current iteration's model
            print('\n', ilmax_fit.params)
            print(ilmax_fit.summary())
            print(ilmax_fit.cov_params(), '\n')

            # BIC and AIC are used to determine if a model can be reduced
            # they will typically change in the same way
            BIC = ilmax_fit.bic
            AIC = ilmax_fit.aic

            # warn the user if they don't change in the same way
            if AIC < self.old_AIC and BIC >= self.old_BIC:
                warnings.warn("AIC and BIC do not agree for removal of parameter: " +
                            '\'' + column_names[self.candidate_params[0]] + '\'', stacklevel=2)

                print("AIC and BIC do not agree for removal of parameter: " +
                      '\'' + column_names[self.candidate_params[0]] + '\'')

            if BIC < self.old_BIC:
                self.best_model = ilmax
                self.best_model_fit = ilmax_fit

                self.excluded_params.append(self.candidate_params[0])

                # update excluded parameter names
                self.excluded_param_names = []

                for i in range(len(column_names)):
                    if i in self.excluded_params:
                        self.excluded_param_names.append(column_names[i])

                # warn the user if they don't change in the same way
                if AIC >= self.old_AIC:
                    warnings.warn("AIC and BIC do not agree for removal of parameter: " +
                            '\'' + column_names[self.candidate_params[0]] + '\'', stacklevel=2)

                    print("AIC and BIC do not agree for removal of parameter: " +
                          '\'' + column_names[self.candidate_params[0]] + '\'')

                # update old BIC and AIC
                self.old_BIC = BIC
                self.old_AIC = AIC

            # move on to next candidate parameter
            self.candidate_params = np.delete(self.candidate_params, 0)

        # final best model
        print('\n', "Excluded parameters: ", self.excluded_param_names)
        print(self.best_model_fit.params)
        print(self.best_model_fit.summary())
        print(self.best_model_fit.cov_params())

    # reaction coordinate from thetaB for outcomes
    def get_outcome_thetaB(self, outcomes):
        outcome_thetaB = []

        outcome_data = [[] for x in range(len(outcomes[0]))]

        # append outcome observations
        for i in range(len(outcomes[0])):
            for j in range(len(outcomes)):
                # if the current parameter is significant
                if j not in self.excluded_params:
                    outcome_data[i].append(outcomes[j][i])

        for i in range(len(outcome_data)):
            outcome_thetaB.append(thetaB(outcome_data[i], self.best_model_fit.params))

        return outcome_thetaB

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

# Performs inertial likelihood maximization on a set of data
class ILMax(GenericLikelihoodModel):
    def __init__(self, endog, exog,
                 num_A_outcomes, num_B_outcomes,
                 num_q_params, num_qdot_params,
                 **kwds):

        self.num_A_outcomes = num_A_outcomes
        self.num_B_outcomes = num_B_outcomes
        self.num_q_params = num_q_params
        self.num_qdot_params = num_qdot_params

        super(ILMax, self).__init__(endog, exog, **kwds)

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

    def fit(self, start_params=None, method='bfgs', maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.insert(0, 'alpha0')
        if start_params == None:
            # Reasonable starting values
            start_params = np.ones(self.exog.shape[1] + 1)
        return super(ILMax, self).fit(start_params=start_params, method=method,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)

# plots the thetaB's of simplified model
def plot_model_thetaB_sigmoid(outcome_filename, bin_width=0.1):
    ilmax = ILMaxModel(outcome_filename)

    a_thetaB = ilmax.get_outcome_thetaB(ilmax.A_data)
    b_thetaB = ilmax.get_outcome_thetaB(ilmax.B_data)

    a_thetaB = sorted(a_thetaB)
    b_thetaB = sorted(b_thetaB)

    plot_thetaB_sigmoid(a_thetaB, b_thetaB, bin_width=bin_width, title=outcome_filename)

#excluding sigmoid fit until decided it's important
# # sigmoid shape for domain and range of [0, 1]
# # beta can be varied to change the slope of the sigmoid
# def sigmoid(x, beta=3):
#     for i in range(len(x)):
#         x[i] = 1 / (1 + pow((x[i]/((1-x[i]))),-beta))
#
#     return x

# sigmoid plotting of thetaB data with a specified bin size
def plot_thetaB_sigmoid(a_thetaB, b_thetaB, bin_width=0.1, title="UNTITLED"):
    # convert list of floats to a list of tuples of the form (float, outcome)
    for i in range(len(a_thetaB)):
        a_thetaB[i] = (a_thetaB[i], 'A')

    for i in range(len(b_thetaB)):
        b_thetaB[i] = (b_thetaB[i], 'B')

    # combine the lists
    thetaB = a_thetaB + b_thetaB
    # sort (sorts based on first tuple element in ascending order by default)
    thetaB.sort()

    num_going_to_B = 0

    bins = np.zeros(int(1/bin_width))

    # start at bin with smallest thetaB val
    curr_bin = int(thetaB[0][0] / bin_width)
    bin_tot = 0

    for i in range(len(thetaB)):
        bin_tot += 1
        if thetaB[i][0] > (curr_bin + 1) * bin_width:
            bins[curr_bin] = num_going_to_B / bin_tot
            num_going_to_B = 0
            bin_tot = 1
            curr_bin += 1

        if thetaB[i][1] == 'B':
            num_going_to_B += 1

    bins[curr_bin] = num_going_to_B / bin_tot

    x_pos = np.arange(0, 1, bin_width)

    plt.bar(x_pos, bins, align='edge', width=bin_width, alpha=0.5)
    plt.xticks(x_pos)
    plt.xlabel("thetaB")
    plt.ylabel('Fraction going to B')
    plt.title(title)

    #excluding sigmoid fit until decided it's important
    # # plotting comparison sigmoid
    # # linespace generate an array from start and stop value
    # x = np.linspace(0, 1, 100)
    # y = np.linspace(0, 1, 100)
    # y = sigmoid(y)
    #
    # # prepare the plot, associate the color r(ed) or b(lue) and the label
    # plt.plot(x, y, 'r')

    plt.show()
