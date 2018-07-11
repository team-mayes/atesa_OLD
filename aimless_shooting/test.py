# Test script to check my work on this code.

import sys

min_dist = 3

eligible = ['one.his_1','three.his_3','four.his_2']
historyfiles = ['one.his','two.his']

def handle_min_dist(indices, history_lines):
    if len(indices) >= 2:
        for i in range(len(indices) - 1):
            if indices[i + 1] - indices[i] < min_dist:
                try:
                    eligible.remove(history_lines[indices[i + 1]].split(' ')[0])
                    del indices[i + 1]
                    return True
                except ValueError:
                    sys.exit('Error: I broke something in t')
    return False


if min_dist > 1:  # only bother if min_dist is greater than one
    for filename in eligible:
        for history_filename in historyfiles:
            if history_filename in filename:
                history_lines = open(history_filename, 'r').readlines()
                indices = []  # initialize indices of matches
                index = 0  # initialize index to keep track
                for history_line in history_lines:
                    if history_line.split(' ')[0] in eligible:
                        indices.append(index)
                    index += 1
                cont = True
                while cont == True:  # to keep my indices from getting tangled up by deletions
                    cont = handle_min_dist(indices, history_lines)

print(eligible)
