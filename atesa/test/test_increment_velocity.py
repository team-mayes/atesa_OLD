import math
import fileinput
import re
import sys
import shutil

filename = '1.3_1.4_2.2_2.6.rst7_1_1_init_fwd.rst'

# Modified from revvels() to increment coordinate values by velocites, rather than reversing velocities.
# Returns the name of the newly-created coordinate file
byline = open(filename).readlines()
pattern = re.compile('[-0-9.]+')  # regex to match numbers including decimals and negatives
pattern2 = re.compile(
    '\s[-0-9.]+')  # regex to match numbers including decimals and negatives, with one space in front
n_atoms = pattern.findall(byline[1])[0]  # number of atoms indicated on second line of .rst file
offset = 2  # appropriate for n_atoms is odd; offset helps avoid modifying the box line
if int(n_atoms) % 2 == 0:  # if n_atoms is even...
    offset = 1  # appropriate for n_atoms is even

shutil.copyfile(filename, 'temp.rst')
for i, line in enumerate(fileinput.input('temp.rst', inplace=1)):
    if int(n_atoms) / 2 + 2 > i >= 2:
        newline = line
        coords = pattern2.findall(newline)  # line of coordinates
        vels = pattern2.findall(byline[i + int(math.ceil(int(n_atoms) / 2))])  # corresponding velocities
        for index in range(len(coords)):
            length = len(coords[index])  # length of string representing this coordinate
            newline = newline.replace(coords[index], str(float(coords[index]) + float(vels[index]))[0:length])
        sys.stdout.write(newline)
    else:
        sys.stdout.write(line)

        # if i >= int(n_atoms) / 2 + 2 and i <= int(n_atoms) + offset:    # if this line is a velocity line
        #     newline = line
        #     for vel in pattern2.findall(newline):
        #         if '-' in vel:
        #             newline = newline.replace(vel, '  ' + vel[2:], 1)   # replace ' -magnitude' with '  magnitude'
        #         else:
        #             newline = newline.replace(vel, '-' + vel[1:], 1)    # replace ' magnitude' with '-magnitude'
        #     sys.stdout.write(newline)
        # else:  # if not a velocity line
        #     sys.stdout.write(line)
