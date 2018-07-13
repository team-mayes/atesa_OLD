import unittest

from ilmax import ILMaxModel

class TestILMaxModel(unittest.TestCase):
    def test_loglike(self):
        print("\nTesting ILMaxModel.best_model_fit()... \n")

        ilmax_model = ILMaxModel()

        file_obj = open("chosen_start_params-test.txt", 'w')

        print("Excluded parameters: ", ilmax_model.excluded_param_names, file=file_obj)
        print('\n', ilmax_model.best_model_fit.params, file=file_obj)
        print('\n', ilmax_model.best_model_fit.summary(), file=file_obj)
        print('\n', ilmax_model.best_model_fit.cov_params(), file=file_obj)

        file_obj.close()


if __name__ == '__main__':
    unittest.main()