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
from statsmodels.base.model import GenericLikelihoodModel


# ILMaxModel
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
    def __init__(self, outcome_filename="outcomes-test.txt", **kwds):
        self.candidate_params = []
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

        # an array of the indices to the parameters
        self.candidate_params = np.arange(-1, num_q_params)

        # iterate over potential parameters
        while len(self.candidate_params) > 0:
            exog_data = [ [] for x in range(num_A_outcomes + num_B_outcomes) ]

            # append A outcomes
            for i in range(num_A_outcomes):
                for j in range(num_params):
                    # if the current parameter is significant
                    # and it isn't the current candidate for comparison
                    if j not in self.excluded_params and j != self.candidate_params[0]:
                        exog_data[i].append(A_data[j][i])

            # append B outcomes
            for i in range(num_B_outcomes):
                for j in range(num_params):
                    # if the current parameter is significant
                    # and it isn't the current candidate for comparison
                    if j not in self.excluded_params and j != self.candidate_params[0]:
                        exog_data[num_A_outcomes + i].append(B_data[j][i])

            new_column_names = []

            for i in range(len(column_names)):
                if i not in self.excluded_params and i != self.candidate_params[0]:
                    new_column_names.append(column_names[i])

            exog = pd.DataFrame(exog_data, columns=new_column_names)
            endog = pd.DataFrame(np.zeros(exog.shape))

            # to avoid errors, endog isn't used in ILMax
            if endog is None:
                endog = np.zeros_like(exog)

            self.ilmax = ILMax(exog=exog, endog=endog,
                               num_A_outcomes=num_A_outcomes,
                               num_B_outcomes=num_B_outcomes,
                               num_q_params=(num_q_params - len(self.excluded_params)),
                               num_qdot_params=num_qdot_params,
                               **kwds)

            self.ilmax_fit = self.ilmax.fit(method='bfgs')

            BIC = self.ilmax_fit.bic
            AIC = self.ilmax_fit.aic

            if BIC < self.old_BIC:
                self.best_model = self.ilmax
                self.best_model_fit = self.ilmax_fit

                # update old BIC and AIC
                self.old_BIC = BIC
                self.old_AIC = AIC

                self.excluded_params.append(self.candidate_params[0])

                # update excluded parameter names
                self.excluded_param_names = []

                for i in range(len(column_names)):
                    if i in self.excluded_params:
                        self.excluded_param_names.append(column_names[i])

                if AIC > self.old_AIC:
                    raise Warning("AIC and BIC do not agree for parameter: ",
                                  column_names[self.candidate_params[0]])

            if AIC < self.old_AIC and BIC > self.old_BIC:
                raise Warning("AIC and BIC do not agree for parameter: ",
                              column_names[self.candidate_params[0]])

            # move on to next candidate parameter
            self.candidate_params = np.delete(self.candidate_params, 0)

# ILMax
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
