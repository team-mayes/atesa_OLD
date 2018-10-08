# from unittest import TestCase
#
#
# class TestReturn_rcs(TestCase):
#     def test_return_rcs(self):
#         self.fail()

literal_ops = True


rc_values = []
for i in range(traj.__len__()):                         # iterate through frames
    op_values = candidatevalues(thread,frame=i).split(' ')  # OP values as a list
    equation = rc_definition
    if literal_ops:
        candidateops = [candidateops]                   # to fix error where candidateops has unexpected format
    for j in reversed(range(len(candidateops[0]))):     # for each candidate op... (reversed to avoid e.g. 'OP10' -> 'candidatevalues(..., 0)0')
        equation = equation.replace('OP' + str(j + 1), 'op_values[j]')
    rc_values.append(eval(equation))
commit_flag = 'False'           # return 'False' if no beads are in bounds
for value in rc_values:
    if rc_min <= value <= rc_max:   # todo: this needs to be changed to the min and max of the given window that this thread is constrained to
        commit_flag = 'True'    # return 'True' if any of the beads are in bounds.
        break
