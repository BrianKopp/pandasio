import unittest
import numpy as np
from pandasio.utils.numpy_compression import round_array_returning_integers
from pandasio.utils.exceptions import *


class TestNumpyFloatRounding(unittest.TestCase):
    def test_rounding(self):
        a = np.array([0.123456789], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(1, i[0])
        i = round_array_returning_integers(a, 2)
        self.assertEqual(12, i[0])
        i = round_array_returning_integers(a, 0)
        self.assertEqual(0, i[0])
        i = round_array_returning_integers(a, 5)
        self.assertEqual(12346, i[0])

        a = np.array([-0.123456789], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(-1, i[0])
        return

    def test_round_errors(self):
        a = np.array([0.1], dtype=np.float32)
        with self.assertRaises(ValueError):
            round_array_returning_integers(a, -1)
        with self.assertRaises(NotIntegerException):
            round_array_returning_integers(a, 0.5)
        return

    def test_good_rounding(self):
        a = np.array([0.45], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(4, i[0])  # numpy rounds to the nearest even when spaced evenly
        a = np.array([-0.45], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(-4, i[0])

        a = np.array([0.5], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(5, i[0])
        a = np.array([-0.5], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(-5, i[0])

        a = np.array([0.44], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(4, i[0])
        a = np.array([-0.44], dtype=np.float32)
        i = round_array_returning_integers(a, 1)
        self.assertEqual(-4, i[0])
        return

if __name__ == '__main__':
    unittest.main()
