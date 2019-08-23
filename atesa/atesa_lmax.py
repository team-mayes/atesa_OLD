from __future__ import division
from scipy import optimize
from scipy.special import erf
import numpy
import random
import math
import re
import sys
import argparse
import itertools
import time


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
        eta_h = str(math.floor(eta/3600))
        eta_m = str(math.floor((eta % 3600) / 60))
        eta_s = str(math.floor((eta % 3600) % 60))
        if len(eta_m) == 1:
            eta_m = '0' + eta_m
        if len(eta_s) == 1:
            eta_s = '0' + eta_s
        eta_str = eta_h + ':' + eta_m + ':' + eta_s
        text = "\r" + message + ": [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 2), status) + " ETA: " + eta_str
    else:
        text = "\r" + message + ": [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def sigmoid_data(params, obs):
    # Returns reaction coordinate value for a given set of parameters and observation
    params = list(params)
    q = params[0] + numpy.dot(params[1:], obs)
    # for index in range(len(obs)):
    #     q += params[index + 1] * obs[index]
    theta = (1 + math.erf(q))/2

    return q


def objective_function(params, A_data, B_data):
    # Returns -1 * (sumA(ln(1 - theta)) + sumB(ln(theta))) where theta = (1 + erf(q))/2 and q is the reaction coordinate
    def erflikefast(arg):
        pl = numpy.ones(len(arg))
        ml = numpy.negative(numpy.ones(len(arg)))
        return numpy.where(arg > 5.7, pl, numpy.where(arg < -5.7, ml, erf(arg)))

    params = list(params)
    qa = params[0] + numpy.inner(params[1:], A_data)
    qb = params[0] + numpy.inner(params[1:], B_data)
    sum = numpy.sum(numpy.log((1 - erflikefast(qa)) / 2)) + numpy.sum(numpy.log((1 + erflikefast(qb)) / 2))
    
    # for observation in A_data:                  # observation is a list of CV values for a given observation
    #     try:
    #         q = params[0] + numpy.dot(params[1:],observation)
    #     except ValueError:
    #         print(params)
    #         print(observation)
    #         sys.exit()
    #     sum += numpy.log(1 - (1 + erflike(q))/2)            # increment log-likelihood by log(1-theta)
    # for observation in B_data:
    #     q = params[0] + numpy.dot(params[1:],observation)
    #     sum += numpy.log((1 + erflike(q))/2)                # increment log-likelihood by log(theta)
    # print sum, suma, params[0]

    return -1 * sum


def main(input_file, kbest=None, fixed=None, include_qdot=False, method='BFGS', running=False, bootstrap=False, guess=None, output_file=None):
    # First, scrape data from input_file as lists in lists A_data and B_data
    A_data = []
    B_data = []
    with open(input_file, 'r') as f:
        pattern = re.compile('\ [0-9.-].*')                     # pattern to scrape values from each line
        for line in f.readlines():
            if line[0] == 'A' or (line[0] == ' ' and line[1] == 'A'):   # tolerates space at beginning, or not
                values = pattern.findall(line)[0].split(' ')    # ['', value1, value2, ..., valueN, '']
                values = list(filter(None, values))             # just the values (without the empty strings)
                A_data.append(numpy.array([float(value) for value in values]))
            elif line[0] == 'B' or (line[0] == ' ' and line[1] == 'B'):
                values = pattern.findall(line)[0].split(' ')    # ['', value1, value2, ..., valueN, '']
                values = list(filter(None, values))             # just the values (without the empty strings)
                B_data.append(numpy.array([float(value) for value in values]))
            else:
                sys.exit('Error: a line in ' + input_file + ' starts with something other than \'A\' or \'B\'')
        f.close()
    A_data = numpy.vstack(A_data)
    B_data = numpy.vstack(B_data)
    # Now we want to reduce all the values by doing value = (value - min(values)/(max(values) - min(values))
    N = len(A_data[0])
    for op_index in range(N):                   # iterate over CVs
        min_val = min(numpy.amin(A_data[:,op_index]), numpy.amin(B_data[:,op_index]))
        max_val = max(numpy.amax(A_data[:,op_index]), numpy.amax(B_data[:,op_index]))
        # for obs_index in range(len(A_data)):    # iterate over A observations
        #     if obs_index == 0:
        #         min_val = A_data[obs_index][op_index]
        #         max_val = A_data[obs_index][op_index]
        #     if A_data[obs_index][op_index] < min_val:
        #         min_val = A_data[obs_index][op_index]
        #     elif A_data[obs_index][op_index] > max_val:
        #         max_val = A_data[obs_index][op_index]
        # for obs_index in range(len(B_data)):    # iterate over B observations
        #     if B_data[obs_index][op_index] < min_val:
        #         min_val = B_data[obs_index][op_index]
        #     elif B_data[obs_index][op_index] > max_val:
        #         max_val = B_data[obs_index][op_index]
        A_data[:,op_index] -= min_val
        A_data[:,op_index] /= (max_val - min_val)
        B_data[:,op_index] -= min_val
        B_data[:,op_index] /= (max_val - min_val)
        # Now that I have the min and max values, go back through and reduce
        # for obs_index in range(len(A_data)):    # iterate over A observations
        #     A_data[obs_index][op_index] = (A_data[obs_index][op_index] - min_val) / (max_val - min_val)
        # for obs_index in range(len(B_data)):    # iterate over B observations
        #     B_data[obs_index][op_index] = (B_data[obs_index][op_index] - min_val) / (max_val - min_val)

    if include_qdot and not N%2 == 0:
        sys.exit('Error: optimization was called with include_qdot == True, but the input file contained an odd number of CVs')

    if not kbest:
        kbest = N
    elif include_qdot:
        kbest = kbest*2     # double parameters to account for corresponding qdot values
    current_best = [objective_function([0 for null in range(N + 1)], A_data, B_data), [-1,-1,-1]]
    if not guess:
        start_params = [0 for null in range(kbest + 1)]
    else:
        start_params = guess    # guess is only ever used during running, so this start_params is always of length kbest (one short)
    optim_result = 'null'

    if fixed:
        if len(fixed) > kbest:
            sys.exit('Error: number of fixed parameters (' + str(len(fixed)) + ') must be less than or equal to number '
                     'of parameters requested (' + str(kbest) + ')')

    # These branches both build a list of combinations to optimize. They use a list comprehension with set.issubset
    # because it's elegant, and more importantly because it keeps memory usage low in cases where going through and
    # removing unnecessary combinations from the full N-choose-kbest list is prohibitively expensive, though that option
    # may well be faster.
    if not running:
        if len(fixed) == int(kbest):
            iterables = [fixed]
        elif not include_qdot:
            iterables = [comb for comb in itertools.combinations(range(N), kbest) if (not fixed or set(fixed).issubset(comb))]
        else:
            iterables = []
            temp_iterables = [comb for comb in itertools.combinations(range(int(N/2)), int(kbest/2)) if (not fixed or set(fixed[:-int(len(fixed)/2)]).issubset(comb))]  # begin with only CVs (no qdot values)
            for comb in temp_iterables:
                temp = list(comb)
                for CV in comb:             # add all the qdot parameters for the included CVs
                    temp.append(int(CV + (N/2)))
                iterables.append(temp)
    else:   # only have to add a single extra dimension to fixed for each item in iterables, so save a lot of time
        if not include_qdot:
            iterables = [fixed+[new] for new in range(N) if not new in fixed]
        else:
            iterables = []
            temp_iterables = [fixed[0:int(len(fixed)/2)]+[new] for new in range(int(N/2)) if not new in fixed]
            for comb in temp_iterables:
                temp = list(comb)
                for CV in comb:             # add all the qdot parameters for the included CVs
                    temp.append(int(CV + (N/2)))
                iterables.append(temp)

    if not iterables:
        sys.exit('Error: the given arguments to LMAX result in an empty list of order parameter combinations to test. '
                 'I see ' + str(N) + ' variables in your input file, of which you\'ve asked me to choose combinations '
                 'of length ' + str(kbest) + ' (this number should be double the value of the k argument if -q is True)'
                 ' containing all of these elements: ' + str(fixed))

    count = 0
    count_to = len(iterables)
    update_progress(0, 'Optimizing ' + str(count_to) + ' combination(s) of CVs')
    speed_values = []

    for comb in iterables:
        if guess:       # have to insert an initial guess for the previously excluded term at the correct position
            start_params = guess
            index = 0   # start at zero to skip start_params[0], which is the constant term
            for cv in comb:
                index += 1
                if cv not in fixed:
                    start_params = start_params[:index] + [0] + start_params[index:]    # insert value for new CV
        t = time.time()
        this_A = []
        this_B = []
        for index in comb:              # produce kbest-by-len(A_data) matrices (list of lists) for the selected CVs
            this_A.append([obs[index] for obs in A_data])
            this_B.append([obs[index] for obs in B_data])
        this_A = list(map(list, zip(*this_A)))      # transpose the matrices to get desired format
        this_B = list(map(list, zip(*this_B)))
        this_result = optimize.minimize(objective_function,numpy.asarray(start_params),(this_A,this_B),
                                        method=method,options={"disp": False, "maxiter": 20000*(kbest + 1)})
                                       # jac='3-point',hess='3-point')
        if this_result.fun < current_best[0]:
            current_best = [this_result.fun, list(comb)]
            optim_result = this_result
        speed_values.append(time.time() - t)
        speed = numpy.mean(speed_values)
        count += 1
        eta = (count_to - count) * speed
        update_progress(count/count_to, 'Optimizing ' + str(count_to) + ' combination(s) of CVs', eta)
    if optim_result == 'null':
        sys.exit('Error: Unable to find result better than null solution.')
    if not output_file:
        print(str(optim_result.message) + '\n')
        print('Function value and included parameter(s) (one-indexed): ' + '%.4f'%(current_best[0]) + ',' + ''.join((' CV' + str(current_best[1][i] + 1)) for i in range(len(current_best[1]))) + '\n')
        print('Corresponding coefficients (starting with constant term): ' + str([float('%.4f'%(item)) for item in optim_result.x]) + '\n')
        if method.lower() == 'bfgs':
            print('Standard error of each term: ' + str([float('%.4f'%(numpy.sqrt(item))) for item in numpy.diag(optim_result.hess_inv)]) + '\n')
            print('Errors as fractions: ' + str([abs(float('%.4f'%(item))) for item in numpy.divide([numpy.sqrt(item) for item in numpy.diag(optim_result.hess_inv)], optim_result.x)]) + '\n')
            print(optim_result.hess_inv)    # todo: gives different results for same optimization depending on whether it was arrived at via running?
    elif not running:
        open(output_file, 'w').write(str(optim_result.message) + '\n')
        open(output_file, 'a').write('Function value and included parameter(s): ' + str(current_best[0]) + ',' + ''.join((' CV' + str(current_best[1][i] + 1)) for i in range(len(current_best[1]))) + '\n')
        open(output_file, 'a').write('Corresponding coefficients (starting with constant term): ' + str(optim_result.x) + '\n')
        open(output_file, 'a').write('Final RC: ' + str(optim_result.x[0]) + ''.join(' + ' + str(optim_result.x[i+1]) + '*CV' + str(current_best[1][i] + 1) for i in range(len(current_best[1]))) + '\n')
        if method.lower() == 'bfgs':
            open(output_file, 'a').write('Standard error of each term: ' + str([float('%.4f'%(numpy.sqrt(item))) for item in numpy.diag(optim_result.hess_inv)]) + '\n')
            open(output_file, 'a').write('Errors as fractions: ' + str([abs(float('%.4f'%(item))) for item in numpy.divide([numpy.sqrt(item) for item in numpy.diag(optim_result.hess_inv)], optim_result.x)]) + '\n')
        open(output_file, 'a').close()

    output_dict = {}        # initialize dictionary to eventually return (unused if not bootstrap and not running)

    if bootstrap:
        binwidth = 0.5
        bins = [(i - 1) * binwidth + (3/2)*binwidth for i in range(-15, 15)]    # 30 bins of width 0.5 centered on 0
        zero_index = bins.index(-binwidth/2)    # index of left boundary of bin centered on 0

        this_A = []
        this_B = []
        for index in current_best[1]:
            this_A.append([obs[index] for obs in A_data])
            this_B.append([obs[index] for obs in B_data])
        this_A = list(map(list, zip(*this_A)))  # transpose the matrices to get desired format
        this_B = list(map(list, zip(*this_B)))

        A_results = []      # list of RC values for A observations
        for obs in this_A:  # iterate over A observations
            A_results.append(sigmoid_data(optim_result.x, obs))
        B_results = []      # list of RC values for B observations
        for obs in this_B:  # iterate over B observations
            B_results.append(sigmoid_data(optim_result.x, obs))
        hist_result = numpy.histogram(A_results + B_results, bins)  # this step just to bin, not the final histogram

        subsampleA = []  # subsets of all observations, for bootstrapping
        subsampleB = []
        for result in A_results:
            if hist_result[1][zero_index] <= result < hist_result[1][
                        zero_index + 1]:  # if this observation is in the zero-centered bin
                subsampleA.append(result)
        for result in B_results:
            if hist_result[1][zero_index] <= result < hist_result[1][zero_index + 1]:
                subsampleB.append(result)

        count_ratio = []
        for bootstrap_iteration in range(bootstrap):
            this_subsampleA = numpy.random.choice(subsampleA, len(subsampleA), True)  # subsample of full size with replacement
            print(this_subsampleA)  # todo: wait so hold on; this won't work because it will always produce the same number of observations going to A and B, respectively; just mixing around their RC values, which doesn't matter at all. What I really need to subsample from is the full population of observations
            this_subsampleB = numpy.random.choice(subsampleB, len(subsampleB), True)

            A_count = 0
            B_count = 0
            for result in this_subsampleA:
                if hist_result[1][zero_index] <= result < hist_result[1][zero_index + 1]:   # if this observation is in the zero-centered bin
                    A_count += 1
            for result in this_subsampleB:
                if hist_result[1][zero_index] <= result < hist_result[1][zero_index + 1]:
                    B_count += 1
            if A_count or B_count:  # if there is data in this bin
                count_ratio.append(B_count / (A_count + B_count))

        stdev = numpy.std(count_ratio)
        avg = numpy.average(count_ratio)
        output_dict.update({'bootstrap': stdev})
        print('Bootstrapped avg of p_B at RC = 0: ' + str(avg))
        print('Bootstrapped stdev of p_B at RC = 0: ' + str(stdev))
        print(count_ratio)
        if not running: # if not running, return now; otherwise, the 'if running' block will return the bootstrap result
            return output_dict  # todo: this currently precludes the output to files two blocks below

    if running:
        if not output_file:
            fname = 'running.out'
        else:
            fname = output_file
        with open(fname, 'a') as f:
            f.write('\n' + str(optim_result.message) + '\n')
            f.write('Function value and included parameter(s): ' + str(current_best[0]) + ',' + ''.join((' CV' + str(current_best[1][i] + 1)) for i in range(len(current_best[1]))) + '\n')
            f.write('Corresponding coefficients (starting with constant term): ' + str(optim_result.x) + '\n')
            f.write('Final RC: ' + str(optim_result.x[0]) + ''.join(' + ' + str(optim_result.x[i+1]) + '*CV' + str(current_best[1][i] + 1) for i in range(len(current_best[1]))) + '\n')
            if method.lower() == 'bfgs':
                f.write('Standard error of each term: ' + str([float('%.4f' % (numpy.sqrt(item))) for item in numpy.diag(optim_result.hess_inv)]) + '\n')
                f.write('Errors as fractions: ' + str([abs(float('%.4f' % (item))) for item in numpy.divide([numpy.sqrt(item) for item in numpy.diag(optim_result.hess_inv)], optim_result.x)]) + '\n')
            f.close()
        output_dict.update({'current_best': current_best[1]})
        output_dict.update({'guess': list(optim_result.x)})
        # return [current_best[1],list(optim_result.x)[:-int(len(optim_result.x)/2)]] # old
        return output_dict

    # This section only runs if running == False and output_file == False
    if not output_file:
        with open('lmax.out', 'w') as f:
            f.write(str(optim_result.message) + '\n')
            f.write('Function value and included parameter(s): ' + str(current_best[0]) + ' ' + str(current_best[1]) + '\n')
            f.write('Corresponding coefficients (starting with constant term): ' + str(optim_result.x) + '\n')
            if method.lower() == 'bfgs':
                f.write('Standard error of each term: ' + str([float('%.4f' % (numpy.sqrt(item))) for item in numpy.diag(optim_result.hess_inv)]) + '\n')
                f.write('Errors as fractions: ' + str([abs(float('%.4f' % (item))) for item in numpy.divide([numpy.sqrt(item) for item in numpy.diag(optim_result.hess_inv)], optim_result.x)]) + '\n')
            f.close()
        with open('reduced_A.out', 'w') as f:
            for obs in A_data:
                f.write(str(obs) + '\n')
            f.close()
        with open('reduced_B.out', 'w') as f:
            for obs in B_data:
                f.write(str(obs) + '\n')
            f.close()
        with open('committor_probability.out', 'w') as f:
            n_bins = 30
            f.write('Theta value (bin size = ' + str(1/n_bins) + ') vs. Probability of commitment to state B\n')

            this_A = []
            this_B = []
            for index in current_best[1]:
                this_A.append([obs[index] for obs in A_data])
                this_B.append([obs[index] for obs in B_data])
            this_A = list(map(list, zip(*this_A)))  # transpose the matrices to get desired format
            this_B = list(map(list, zip(*this_B)))

            A_results = []
            for obs in this_A:              # iterate over A observations
                A_results.append(sigmoid_data(optim_result.x, obs))
            B_results = []
            for obs in this_B:              # iterate over B observations
                B_results.append(sigmoid_data(optim_result.x, obs))
            hist_result = numpy.histogram(A_results + B_results, n_bins)  # this step just to bin, not the final histogram
            for bin_index in range(len(hist_result[0])):
                A_count = 0
                B_count = 0
                for result in A_results:
                    if hist_result[1][bin_index] <= result < hist_result[1][bin_index+1]:
                        A_count += 1
                for result in B_results:
                    if hist_result[1][bin_index] <= result < hist_result[1][bin_index+1]:
                        B_count += 1
                if A_count or B_count:  # if there is data in this bin
                    count_ratio = B_count / (A_count + B_count)
                else:
                    count_ratio = 'NaN'
                f.write(str(numpy.mean([hist_result[1][bin_index+1],hist_result[1][bin_index]])) + ' ' + str(count_ratio) + '\n')
            f.close()


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Perform LMAX on the given input data')
    parser.add_argument('-i', metavar='input_file', type=str, nargs=1, default='as.out',
                        help='input filename. Default=\'as.out\'')
    parser.add_argument('-k', metavar='kbest', type=int, nargs=1, default=[0],
                        help='number of CVs to include in RC. Default=all')
    parser.add_argument('-f', metavar='fixed', type=int, nargs='*', default=None,
                        help='CVs to require inside the RC. Default=none')
    parser.add_argument('-q', metavar='include_qdot', type=str2bool, nargs=1, default=[False],
                        help='whether or not to include CV rates of change (2nd half of each observation in input file)'
                             ' in optimization. Default=False')
    parser.add_argument('-m', metavar='method', type=str, nargs=1, default=['bfgs'],
                        help='optimization method; see scipy.optimize.minimize documentation for details. Default=bfgs')
    parser.add_argument('--running', metavar='running', type=int, nargs='*', default=[0],
                        help='if > 0, runs from k = 1 to running using the previously obtained k-1 results as the '
                             'argument for f. Ignores the arguments passed for k and f. Default=None')
    parser.add_argument('--bootstrap', metavar='bootstrap', type=int, nargs=1, default=[0],
                        help='if not 0, the returned dictionary object will include the standard deviation of the error'
                             ' of the p_B value at RC = 0 obtained from \'bootstrap\' iterations of bootstrapping, with'
                             ' the key \'bootstrap\'. Default=0')   # todo: Bootstrapping currently broken; might remove it!
    parser.add_argument('--output_file', metavar='output_file', type=str, nargs=1, default=[''],
                        help='Prints output to a new file whose name is given with this argument, instead of directly '
                             'to the terminal. Default=None')
    arguments = vars(parser.parse_args())  # Retrieves arguments as a dictionary object

    if arguments.get('bootstrap')[0] < 0:
        sys.exit('Error: likelihood maximization was called with bootstrap < 0 (must be >= 0).')
    elif arguments.get('k')[0] < 0:
        sys.exit('Error: likelihood maximization was called with kbest < 0 (must be >= 0).')
    # elif arguments.get('f')[0] < 0:
    #     sys.exit('Error: likelihood maximization was called with fixed < 0 (must be >= 0).')

    # print(arguments.get('i')[0])
    # print(arguments.get('k')[0])
    # print(arguments.get('f'))
    # print(arguments.get('q')[0])
    # print(arguments.get('m'))
    # print(arguments.get('running')[0])
    # print(arguments.get('bootstrap')[0])
    if not arguments.get('running')[0]:
        main(arguments.get('i')[0], arguments.get('k')[0], arguments.get('f'), arguments.get('q')[0], arguments.get('m')[0], arguments.get('running')[0], arguments.get('bootstrap')[0], output_file=arguments.get('output_file')[0])
    else:
        open('running.out', 'w').close()    # initialize output file
        # Run increasingly higher-dimensional optimization using the selected parameters from the previous run as the
        # "fixed" variables and the solution to the previous run as the initial guess for those variables.
        f = []
        g = []
        start_k = 0
        
        for k in range(arguments.get('running')[0]):
            if k >= start_k:
                output_dict = main(arguments.get('i')[0], k+1, f, arguments.get('q')[0], arguments.get('m')[0], arguments.get('running')[0], arguments.get('bootstrap')[0], g, output_file=arguments.get('output_file')[0])
                f = output_dict['current_best']
                g = output_dict['guess']
