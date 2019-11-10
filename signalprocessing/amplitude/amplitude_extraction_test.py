import unittest
import numpy as np

from amplitude_extraction import avg_band_amplitude


class AmplitudeExtractionTest(unittest.TestCase):
    def test_avg_band_amplitude_returns_correct_avg_amplitude_within_selected_limits(self):
        avg_amplitude = avg_band_amplitude(np.array(range(0, 30)), 10, 20)
        self.assertEqual(15, avg_amplitude)


if __name__ == '__main__':
    unittest.main()
