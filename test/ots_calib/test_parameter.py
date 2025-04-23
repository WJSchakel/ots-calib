'''
Tests.

@author: wjschakel
'''
import unittest

from ots_calib.parameters import Parameter, Parameters

# TODO: more testing


class ParameterTest(unittest.TestCase):

    def testParameter(self):

        # Static bounds
        parameter_a = Parameter('a', 5.0, 0.0, 10.0)
        parameters = Parameters((parameter_a,))
        self.assertRaises(KeyError, parameters.set_value, 'b', 5.0)
        self.assertRaises(ValueError, parameters.set_value, 'a', -1.0)
        self.assertRaises(ValueError, parameters.set_value, 'a', 11.0)

        # Resetting
        parameters.set_value('a', 3.0)
        self.assertEqual(parameters.get_value('a'), 3.0, 'Parameter should be 3.0')
        parameters.reset_value('a')
        self.assertEqual(parameters.get_value('a'), 5.0, 'Reset parameter should be 5.0')
        parameters.reset_value('a')  # should not raise exception although already reset
        parameters.reset_value('b')  # should not raise exception although not present

        # Dynamic bounds
        parameter_a = Parameter('a', 2.5, 0.0, 'b')
        parameter_b = Parameter('b', 7.5, 'a', 10.0)
        parameters = Parameters((parameter_a, parameter_b))
        self.assertRaises(ValueError, parameters.set_value, 'a', 8.0)
        self.assertRaises(ValueError, parameters.set_value, 'b', 2.0)

        # Content correct
        self.assertEqual(parameters.get_ids(), ['a', 'b'], 'Parameter ids not equal')
        self.assertEqual(parameters.get_values(), [2.5, 7.5], 'Parameter values not correct')

        # Initial values
        self.assertEqual(parameters.get_parameter('a').get_initial_value(), 2.5,
                         'Parameter has incorrect initial value')
        self.assertEqual(parameters.get_parameter('b').get_initial_value(), 7.5,
                         'Parameter has incorrect initial value')
        parameters.set_value('a', 3.0)
        parameters.set_value('b', 7.0)
        self.assertEqual(parameters.get_values(), [3.0, 7.0], 'Parameter values not correct')
        parameters.set_initial()
        self.assertEqual(parameters.get_values(), [2.5, 7.5], 'Parameter values not initial')


if __name__ == "__main__":
    unittest.main()
