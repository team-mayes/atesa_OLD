import pymbar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import random

def main(input_file,bootstrap=True,bootstrapN=0,error=[]):
    # Main function that calls itself recursively once with bootstrap = True to build the error estimate

    overlap = 0 # should be 0.2 if I want to build a new window for each overlapping region

    file = open(input_file, 'r').readlines()
    open(input_file, 'r').close()

    windows = []    # nested list of format [[lower0,upper0],[lower1,upper1],...]
    data = []       # nested list with indices corresponding windows, format [[x00,x01,...],[x10,x11,...],...]
    alldata = []    # simple list [x00,x01,...x0N,x10,x11,...]
    # Determine the window boundaries
    for line in file:
        line = line.strip('\n')
        if not line[0] == 'L':  # "Lower...", want to skip the header line
            split = line.split(' ')
            if [float('%.3f' % (float(split[0]) + overlap)), float('%.3f' % (float(split[1]) - overlap))] not in windows:
                windows.append([float('%.3f' % (float(split[0]) + overlap)), float('%.3f' % (float(split[1]) - overlap))])
                data.append([])
                if overlap > 0:
                    if [float('%.3f' % (float(split[0]))), float('%.3f' % (float(split[0]) + overlap))] not in windows:
                        windows.append([float('%.3f' % (float(split[0]))), float('%.3f' % (float(split[0]) + overlap))])
                        data.append([])
                    if [float('%.3f' % (float(split[1]) - overlap)), float('%.3f' % (float(split[1])))] not in windows:
                        windows.append([float('%.3f' % (float(split[1]) - overlap)), float('%.3f' % (float(split[1])))])
                        data.append([])
                # temp_windows = windows[-3:]
                # temp_windows.sort(key=lambda x: x[0])
                # print(temp_windows)

    windows.sort(key=lambda x: x[0])    # need to be sorted for building the PMF
    if overlap > 0:
        windows = windows[1:-1]         # remove empty windows outside boundaries
    for line in file:
        line = line.strip('\n')
        if not line[0] == 'L':          # "Lower...", want to skip the header line
            split = line.split(' ')
            if overlap > 0:
                local_index = -1
                for window in windows:
                    local_index += 1
                    if window[0] <= float(split[2]) < window[1]:
                        data[local_index].append(float(split[2]))
                        break
            else:
                data[windows.index([float('%.3f' % float(split[0])),float('%.3f' % float(split[1]))])].append(float(split[2]))
            alldata.append(float(split[2]))

    # windows = windows[3:-1]             #todo: ### REMOVE THESE LINES BEFORE PUBLISHING ###
    # data = data[3:-1]

    if bootstrap:
        for i in range(len(windows)):
            data[i] = random.sample(data[i],bootstrapN)

    nbins = 4
    left_boundary = 0
    fullPMF = []
    fullRC = []
    allprobs = []
    if not bootstrap:
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        fig = plt.figure()
        ax4 = fig.add_subplot(111)
        error_index = 0
    for index in range(len(windows)):
        #nbins = max([int(np.ceil(len(data[index])/60)),4])
        probs = [0 for null in range(nbins)]
        thismin = min(data[index])
        thismax = max(data[index])
        count = 0
        for value in data[index]:
            reduced = (value - thismin)/(thismax - thismin)     # reduce to between 0 and 1
            local_index = int(np.floor(reduced*nbins))
            if local_index == nbins:
                local_index -= 1      # in case where reduced == 1
            try:
                probs[local_index] += 1
            except IndexError:
                sys.exit(str(local_index))
            count += 1
        for i in range(len(probs)):
            probs[i] = probs[i]/len(data[index])
        local_index = 0
        U = [0 for null in range(nbins)]
        RC_values = np.linspace(windows[index][0], windows[index][1], nbins)
        offset = 0
        for prob in probs:
            if local_index == 0 and not bootstrap:
                offset = -0.596 * np.log(prob)
            if not prob == 0:
                U[local_index] = -0.596 * np.log(prob) - offset
            local_index += 1
        if index == 0 or bootstrap: # turn off boundary value matching during bootstrapping to avoid propagating errors in this step into PMF error in final step
            left_boundary = 0
        else:
            f1 = (RC_values[0] - boundary_values[3]) / (boundary_values[2] - boundary_values[3])     # fraction between 0 and 1 corresponding to distance first point of next window falls between last two points of previous window
            left_boundary1 = ((boundary_values[0] - boundary_values[1]) * f1) + boundary_values[1]   # shift to lower next window to intersect last one
            f2 = (boundary_values[2] - RC_values[0]) / (RC_values[1] - RC_values[0])
            left_boundary2 = boundary_values[0] - ((U[1] - U[0]) * f2) + U[0]                    # shift to lower next window so that previous one intersects it
            left_boundary = np.mean([left_boundary1,left_boundary2])                                # average of shift amounts
        for Uindex in range(len(U)):
            U[Uindex] += left_boundary
        boundary_values = [U[-1], U[-2], RC_values[-1], RC_values[-2]]  # [x2, x1, r2, r1]
        fullPMF += U
        if not bootstrap:
            ax0.errorbar(RC_values,U,error[error_index:error_index+len(U)])
            fullRC += list(RC_values)
            error_index += len(U)
            fig.canvas.draw()
            nextcolor = list(colors.to_rgb(next(ax4._get_patches_for_fill.prop_cycler).get('color'))) + [0.75]
            ax4.bar(np.linspace(windows[index][0],windows[index][1],len(probs)),probs,width=0.2,color=nextcolor)
            fig.canvas.draw()
    if not bootstrap:
        pass
        plt.show()
    else:
        return fullPMF

    # Section for smoothing the PMF into a single continuous line
    tempPMF = []
    tempRC = []
    tempErr = []
    i = 0
    while i < len(fullPMF):
        if (i+1)%nbins == 0 and i+1 < len(fullPMF):
            tempPMF.append(np.mean([fullPMF[i], fullPMF[i + 1]]))
            tempRC.append(np.mean([fullRC[i], fullRC[i + 1]]))
            tempErr.append(np.mean([error[i], error[i + 1]]))
            i += 1  # skip next point
        else:
            tempPMF.append(fullPMF[i])
            tempRC.append(fullRC[i])
            tempErr.append(error[i])
        i += 1
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax1.errorbar(tempRC,tempPMF,tempErr)
    tempErr = list(reversed(tempErr))
    tempPMF = list(reversed(tempPMF))
    tempRC = list(reversed([-1*RC for RC in tempRC]))
    with open(input_file + '.data', 'w') as f:
        for i in range(len(tempRC)):
            f.write(str(tempRC[i]) + ' ' + str(tempPMF[i]) + ' ' + str(tempErr[i]) + '\n')
    ax1.plot(tempRC,tempPMF,color='#0072BD',lw=2)
    plt.fill_between(np.asarray(tempRC), np.asarray(tempPMF) - np.asarray(tempErr), np.asarray(tempPMF) + np.asarray(tempErr),
                   alpha=0.5, facecolor='#0072BD')
    plt.ylabel('Free Energy (kcal/mol)', weight='bold')
    plt.xlabel('Reaction Coordinate', weight='bold')
    fig.canvas.draw()
    plt.show()

    # Section for writing WHAM input files, if desired
    # open('eps.meta','w').close()
    # for index in range(len(windows)):
    #     open('eps' + str(index) + '.data','w').close()
    #     count = 0
    #     for value in data[index]:
    #         count += 1
    #         open('eps' + str(index) + '.data','a').write(str(count) + ' ' + str(value) + '\n')
    #     open('eps' + str(index) + '.data', 'a').close()
    #     open('eps.meta','a').write('eps' + str(index) + '.data ' + str(np.mean(windows[index])) + ' 0\n')

    # Plot a normalized histogram of each window
    # fig = plt.figure()
    # ax2 = fig.add_subplot(111)
    # for index in range(len(windows)):
    #     # To build nextcolor, we access the default color cycle object and then append an alpha value
    #     nextcolor = list(colors.to_rgb(next(ax2._get_patches_for_fill.prop_cycler).get('color'))) + [0.75]
    #     n, bins, rectangles = ax2.hist(np.asarray(data[index]), nbins, normed=True, color=nextcolor)
    #     fig.canvas.draw()
    # plt.show()

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

if __name__ == '__main__':
    input_file = 'test/a13_eps_results.out'
    bootstrapN = 25    # number of bootstrap samples to include in each window
    bootstrapCyc = 1000  # number of bootstrapping iterations to perform in error estimation
    if bootstrapCyc:
        means = []          # initialize list of bootstrapping results
        std = []            # initialize list of standard error values
        for i in range(bootstrapCyc):
            means.append(main(input_file,bootstrap=True,bootstrapN=bootstrapN))
            update_progress((i+1)/bootstrapCyc, message='Bootstrapping')
        for j in range(len(means[0])):
            this_window = [u[j] for u in means]                     # j'th value from each bootstrapping iteration
            std.append(np.std(this_window))                         # standard deviation for this window
        main(input_file,bootstrap=False,error=std)
    else:
        main(input_file,bootstrap=False)
