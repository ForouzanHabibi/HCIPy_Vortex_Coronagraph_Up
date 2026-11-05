"""Pure NumPy helper checks; HCIPy is not required for these tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from validate_coronagraph import radial_profile


def test_radial_profile_means():
    r = np.array([0.1, 0.4, 0.6, 1.1])
    intensity = np.array([2., 4., 6., 10.])
    out = radial_profile(intensity, r, np.array([0., .5, 1., 1.5]))
    np.testing.assert_allclose(out, [3., 6., 10.])


def test_radial_profile_empty_bin_nan():
    out = radial_profile([5.], [.2], np.array([0., .5, 1.]))
    assert out[0] == 5
    assert np.isnan(out[1])
