import unittest

from ilmax import ILMaxModel, plot_model_thetaB_sigmoid, plot_thetaB_sigmoid

class TestILMaxModel(unittest.TestCase):
    # tests that the variables are independent of each other by
    # varying the order at which variables are removed from the model
    @unittest.skip
    def test_column_pos(self):
        print("Testing ILMaxModel.best_model_fit() for varying column positions... \n")

        # for this example, var4 should be excluded
        excluded_param_names = [ 'var4' ]

        ilmax1234 = ILMaxModel("outcomes-test1234.txt")
        assert (ilmax1234.excluded_param_names == excluded_param_names)

        ilmax1243 = ILMaxModel("outcomes-test1243.txt")
        assert (ilmax1243.excluded_param_names == excluded_param_names)

        ilmax1423 = ILMaxModel("outcomes-test1423.txt")
        assert (ilmax1423.excluded_param_names == excluded_param_names)

        ilmax4123 = ILMaxModel("outcomes-test4123.txt")
        assert (ilmax4123.excluded_param_names == excluded_param_names)

    # tests that the reaction coordinates are in the range [0,1]
    @unittest.skip
    def test_rc_limits(self):
        print("Testing that reaction coordinates are in the range [0,1]... \n")
        ilmax = ILMaxModel("outcomes-baron-rc-test.txt")

        a_thetaB = ilmax.get_outcome_thetaB(ilmax.A_data)
        b_thetaB = ilmax.get_outcome_thetaB(ilmax.B_data)

        a_thetaB = sorted(a_thetaB)
        b_thetaB = sorted(b_thetaB)

        thetaB = a_thetaB + b_thetaB

        min_thetaB = min(thetaB)
        max_thetaB = max(thetaB)

        assert (min_thetaB >= 0 and max_thetaB <= 1)

    # tests the shape of the rc's
    def test_reaction_coordinates(self):
        print("Testing the shape of the rc's... \n")
        plot_model_thetaB_sigmoid(outcome_filename="outcomes-tucker-sigmoid.txt")
        # add more as needed
        # plot_model_thetaB_sigmoid(outcome_filename="outcomes-tucker-sigmoid.txt")
        # plot_model_thetaB_sigmoid(outcome_filename="outcomes-tucker-sigmoid.txt")

    # tests plot_sigmoid for arbitrary pts
    @unittest.skip
    def test_plot_sigmoid(self):
        print("Testing plot_sigmoid... \n")

        a_rc = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.59]
        b_rc = [0.41, 0.6, 0.7, 0.8, 0.9, 1]

        plot_thetaB_sigmoid(a_rc, b_rc, 1/6)

if __name__ == '__main__':
    unittest.main()