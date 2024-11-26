import pytest
from elects import PointElectrode
from brian2 import *
from numpy.testing import assert_array_equal, assert_allclose
from scipy.signal import square

@pytest.fixture
def sine_elect(axon_morpho: Morphology) -> PointElectrode:
    return PointElectrode(current_amp=-11 * uA, frequency=200 * Hz, rx=1000 * um, ry=1000 * um, rz=500 * um,
                          sigma_ex=0.2 * siemens / meter)


@pytest.fixture
def pulse_elect(axon_morpho: Morphology) -> PointElectrode:
    return PointElectrode(current_amp=-11 * uA, rx=-1000 * umeter, ry=1000 * umeter, rz=-500 * umeter,
                          pulse_width=0.3 * ms, sine_wave=False, sigma_ex=0.2 * siemens / meter, duty_cycle=0.5)

@pytest.fixture
def axon_morpho() -> Morphology:
    internode_length, paranode_length, node_length = 110 * um, 3 * um, 1 * um
    internode_diam, paranode_diam, node_diam = 2 * um, 1.4 * um, 1.4 * um

    axon_morpho = Cylinder(diameter=node_diam, length=node_length)
    axon_morpho.p1 = Cylinder(diameter=paranode_diam, length=paranode_length)
    axon_morpho.p1.i1 = Cylinder(diameter=internode_diam, length=internode_length)
    axon_morpho.p1.i1.p2 = Cylinder(diameter=paranode_diam, length=paranode_length)
    axon_morpho.p1.i1.p2.n2 = Cylinder(diameter=node_diam, length=node_length)
    return axon_morpho


def test_init_exceptions(axon_morpho) -> None:
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
    with pytest.raises(TypeError):
        PointElectrode(current_amp=-11 * uA, frequency=200 * Hz, rx=1000 * um, ry=1000 * um, rz=500 * um,
                       sigma_ex=0.2 * siemens / meter, origin=1, morphology=axon_morpho, node_length=1 * um,
                       internode_length=110 * um, paranode_length=None)
    with pytest.raises(TypeError):
        PointElectrode(current_amp=-11 * uA, rx=-1000 * umeter, ry=1000 * umeter, rz=-500 * umeter,
                       pulse_width=0.3 * ms, sine_wave=False, sigma_ex=0.2 * siemens / meter, duty_cycle=0.5,
                       origin=1, morphology=axon_morpho, node_length=1 * um, internode_length=110 * um,
                       paranode_length=None)
    with pytest.raises(ValueError):
        PointElectrode(current_amp=-11 * uA, frequency=200 * Hz, rx=1000 * um, ry=1000 * um, rz=500 * um,
                       sigma_ex=0.2 * siemens / meter, origin=6, morphology=axon_morpho, node_length=1 * um,
                       internode_length=110 * um, paranode_length=3 * um)
    with pytest.raises(ValueError):
        PointElectrode(current_amp=-11 * uA, rx=-1000 * umeter, ry=1000 * umeter, rz=-500 * umeter,
                       pulse_width=0.3 * ms, sine_wave=False, sigma_ex=0.2 * siemens / meter, duty_cycle=0.5,
                       origin=6, morphology=axon_morpho, node_length=1 * um, internode_length=110 * um,
                       paranode_length=3 * um)
    with pytest.raises(ValueError):
        PointElectrode(current_amp=-11 * uA, rx=-1000 * umeter, ry=1000 * umeter, rz=-500 * umeter,
                       pulse_width=0.3 * ms, sine_wave=False, sigma_ex=0.2 * siemens / meter, duty_cycle=-0.5,
                       origin=1, morphology=axon_morpho, node_length=1 * um, internode_length=110 * um,
                       paranode_length=3 * um)
    with pytest.raises(ValueError):
        PointElectrode(current_amp=-11 * uA, rx=-1000 * umeter, ry=1000 * umeter, rz=-500 * umeter,
                       pulse_width=0.3 * ms, sine_wave=False, sigma_ex=0.2 * siemens / meter, duty_cycle=2,
                       origin=1, morphology=axon_morpho, node_length=1 * um, internode_length=110 * um,
                       paranode_length=3 * um)

def test_elect_mem_distance(sine_elect: PointElectrode, pulse_elect: PointElectrode) -> None:
    assert sine_elect.elect_mem_dist() == 1500 * umeter
    assert pulse_elect.elect_mem_dist() == 1500 * umeter


def test_v_waveform(sine_elect: PointElectrode, pulse_elect: PointElectrode) -> None:
    t = np.arange(0, 3.1e-3, 1e-4) * second

    sine_actual = sine_elect.v_waveform(t)
    sine_desired = ((sine_elect.current_amp * sin(2 * np.pi * sine_elect.frequency * t)) /
                    (4 * np.pi * sine_elect.elect_mem_dist() * sine_elect.sigma_ex))

    pulse_actual = pulse_elect.v_waveform(t)
    pulse_desired = ((pulse_elect.current_amp * square(2 * np.pi * ((pulse_elect.duty_cycle * t) / pulse_elect.pw),
                                                       pulse_elect.duty_cycle)) / (4 * np.pi *
                                                                                   pulse_elect.elect_mem_dist() *
                                                                                   pulse_elect.sigma_ex))
    assert_array_equal(sine_actual, sine_desired)
    assert_array_equal(pulse_actual, pulse_desired)





