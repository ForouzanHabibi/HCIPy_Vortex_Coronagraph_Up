import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from coronagraph_sim import zernike_mode, circular_annulus_average

@pytest.mark.parametrize('mode', ['defocus', 'astigmatism', 'coma'])
def test_finite_zernike(mode):
    x, y = np.meshgrid(np.linspace(-.5,.5,30), np.linspace(-.5,.5,30))
    z = zernike_mode(x,y,mode)
    assert z.shape == x.shape
    assert np.isfinite(z).all()

def test_circular_average():
    x, y = np.meshgrid(np.linspace(-5,5,101), np.linspace(-5,5,101))
    assert circular_annulus_average(np.ones_like(x)*0.1, x,y) == pytest.approx(0.1)

def test_unknown_mode():
    with pytest.raises(ValueError):
        zernike_mode(np.array([0.]), np.array([0.]), 'unknown')
