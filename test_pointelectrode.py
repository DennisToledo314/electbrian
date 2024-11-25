import pytest
from elects import PointElectrode
from brian2 import *

def test_init_exceptions() -> None:
    with pytest.raises(ValueError):
        PointElectrode(current_amp=-11 * uA, frequency=0, rx=1000 * umeter,
                       ry=1000 * umeter, rz=500 * umeter, sigma_ex=0.2 * siemens / meter)
    with pytest.raises(ValueError):
        PointElectrode(current_amp=-11 * uA, pulse_width=0, rx=1000 * umeter,
                       ry=1000 * umeter, rz=500 * umeter, sine_wave=False,
                       sigma_ex=0.2 * siemens / meter)
    with pytest.raises(ValueError):
        PointElectrode(current_amp=-11 * uA, frequency=200 * Hz, rx=1000 * umeter,
                       ry=1000 * umeter, rz=500 * umeter, sigma_ex=0 * siemens / meter)



