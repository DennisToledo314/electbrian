import pytest
from elects import PointElectrode
from brian2 import *
from numpy.testing import assert_array_equal, assert_allclose
from scipy.signal import square

@pytest.fixture
def sine_elect() -> PointElectrode:
    return PointElectrode(current_amp=-11 * uA, frequency=200 * Hz, rx=1000 * um, ry=1000 * um, rz=500 * um,
                          sigma_ex=0.2 * siemens / meter)


@pytest.fixture
def pulse_elect() -> PointElectrode:
    return PointElectrode(current_amp=-11 * uA, rx=-1000 * umeter, ry=1000 * umeter, rz=-500 * umeter,
                          pulse_width=0.3 * ms, sine_wave=False, sigma_ex=0.2 * siemens / meter, duty_cycle=0.5)

@pytest.fixture
def elect_for_testing (axon_morpho: Morphology) -> PointElectrode:
    return PointElectrode(current_amp=-11 * uA, frequency=200 * Hz, rx=1000 * um, ry=1000 * um, rz=500 * um,
                          sigma_ex=0.2 * siemens / meter, origin=2, morphology=axon_morpho, node_length=1 * um,
                          internode_length=110 * um, paranode_length=3 * um)

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


def test_origin_to_mid(elect_for_testing: PointElectrode, axon_morpho: Morphology) -> None:
    internode_length, paranode_length, node_length = 110 * um, 3 * um, 1 * um

    assert_allclose(elect_for_testing.origin_to_mid(-1, 1, paranode_length, axon_morpho),
                    (0.5 * (paranode_length + internode_length)))
    assert_allclose(elect_for_testing.origin_to_mid(1, 3, paranode_length, axon_morpho),
                    (0.5 * (paranode_length + internode_length)))

    assert_allclose(elect_for_testing.origin_to_mid(-1, 0, node_length, axon_morpho),
                    (0.5 * (2 * paranode_length + internode_length + node_length)))
    assert_allclose(elect_for_testing.origin_to_mid(1, 4, node_length, axon_morpho),
                    (0.5 * (2 * paranode_length + internode_length + node_length)))

    with pytest.raises(ValueError):
        elect_for_testing.origin_to_mid(1, 5, paranode_length, axon_morpho)

    with pytest.raises(ValueError):
        elect_for_testing.origin_to_mid(-1, -1, paranode_length, axon_morpho)

    with pytest.raises(ValueError):
        elect_for_testing.origin_to_mid(2, 4, node_length, axon_morpho)

    with pytest.raises(ValueError):
        elect_for_testing.origin_to_mid(-2, 4, node_length, axon_morpho)



def test_v_morpho(elect_for_testing: PointElectrode, axon_morpho: Morphology) -> None:
    internode_length, paranode_length, node_length = 110 * um, 3 * um, 1 * um
    v_actual1 = elect_for_testing.v_morpho(node_length, paranode_length, internode_length, -1, -1, axon_morpho)
    v_actual2 = elect_for_testing.v_morpho(node_length, paranode_length, internode_length, 5, 1, axon_morpho)
    v_actual = {**v_actual1, **v_actual2}
    assert v_actual[1] == v_actual[3]
    assert v_actual[0] == v_actual[4]

def test_v_applied_spatial(elect_for_testing: PointElectrode, axon_morpho: Morphology) -> None:
    internode_length, paranode_length, node_length = 110 * um, 3 * um, 1 * um
    v_actual = elect_for_testing.v_applied_spatial(node_length, paranode_length, internode_length, axon_morpho)
    assert v_actual[1] == v_actual[3]
    assert v_actual[0] == v_actual[4]

    elect_for_testing.origin = 4
    v_test1 = elect_for_testing.v_applied_spatial(node_length, paranode_length, internode_length, axon_morpho)

    elect_for_testing.origin = 0
    v_test2 = elect_for_testing.v_applied_spatial(node_length, paranode_length, internode_length, axon_morpho)

    assert v_test1[0] == v_test2[4]
    assert v_test1[1] == v_test2[3]
    assert v_test1[2] == v_test2[2]

def test_i_applied_spatial() -> None:
    internode_length, paranode_length, node_length = 110 * um, 3 * um, 1 * um
    internode_diam, paranode_diam, node_diam = 2 * um, 1.4 * um, 1.4 * um

    my_morpho = Cylinder(diameter=node_diam, length=node_length)
    my_morpho.p1 = Cylinder(diameter=paranode_diam, length=paranode_length)
    my_morpho.p1.i1 = Cylinder(diameter=internode_diam, length=internode_length)
    my_morpho.p1.i1.p2 = Cylinder(diameter=paranode_diam, length=paranode_length)
    my_morpho.p1.i1.p2.n2 = Cylinder(diameter=node_diam, length=node_length)
    my_morpho.p1.i1.p2.n2.p3 = Cylinder(diameter=paranode_diam, length=paranode_length)
    my_morpho.p1.i1.p2.n2.p3.i2 = Cylinder(diameter=internode_diam, length=internode_length)
    my_morpho.p1.i1.p2.n2.p3.i2.p4 = Cylinder(diameter=paranode_diam, length=paranode_length)
    my_morpho.p1.i1.p2.n2.p3.i2.p4.n3 = Cylinder(diameter=node_diam, length=node_length)
    my_morpho.p1.i1.p2.n2.p3.i2.p4.n3.p5 = Cylinder(diameter=paranode_diam, length=paranode_length)
    my_morpho.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3 = Cylinder(diameter=internode_diam, length=internode_length)
    my_morpho.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6 = Cylinder(diameter=paranode_diam, length=paranode_length)
    my_morpho.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.n4 = Cylinder(diameter=node_diam, length=node_length)

    leak_potential, rest_potential, na_potential, k_potential = -90 * mV, -90 * mV, 50 * mV, -90 * mV
    g_na_f, g_na_p, g_k, g_l_node = 3 * siemens / cm ** 2, 0.01 * siemens / cm ** 2, 0.08 * siemens / cm ** 2, 0.007 * siemens / cm ** 2
    g_l_inter, inter_c, para_c = 0.7e-4 * siemens / cm ** 2, 1.6 * uF / cm ** 2, 1.6 * uF / cm ** 2
    membrane_c, axial_rho = 2 * uF / cm ** 2, 70 * ohm * cm

    eqs = '''
        gNaf: siemens/meter**2
        gNap: siemens/meter**2
        gK : siemens/meter**2
        gL : siemens/meter**2

        Im = -1*(IK+INaf+INap+IL) : amp/meter**2
        INaf = gNaf*(m**3)*h*(v-ENa) : amp/meter**2
        INap = gNap*(mp**3)*(v-ENa) : amp/meter**2
        IK = gK*s*(v-EK) : amp/meter**2
        IL = gL*(v-EL) : amp/meter**2
        i_appl : amp (point current)

        dm/dt = (alpha_m * (1-m) - beta_m * m) : 1
        dh/dt = (alpha_h * (1-h) - beta_h * h) : 1
        dmp/dt = (alpha_mp * (1-mp) - beta_mp * mp) : 1
        ds/dt = (alpha_s * (1-s) - beta_s * s) : 1

        alpha_m = (1.85*(1/ms)*10.3*mV*3.82)/(mV*exprel(-1*(v+21.4*mV)/(10.3*mV))) : Hz
        beta_m = (0.076*(1/ms)*9.16*mV*3.82)/(mV*exprel((v+25.7*mV)/(9.16*mV))) : Hz

        alpha_h = (0.034*(1/ms)*11*mV*6.11)/(mV*exprel((v+112*mV)/(11*mV))) : Hz
        beta_h = (2.3*6.11/ms)/(1+exp(-(v+28.8*mV)/(13.6*mV))) : Hz

        alpha_mp = (0.03*(1/ms)*10.2*mV*3.82)/(mV*exprel(-1*(v+23*mV)/(10.2*mV))) : Hz
        beta_mp = (0.00019*(1/ms)*10*mV*3.82)/(mV*exprel((v+38*mV)/(10*mV))) : Hz

        alpha_s = (0.0138*(1/ms)*9.4*mV*1.12)/(mV*exprel(-1*(v+14*mV)/(9.4*mV))) : Hz
        beta_s = (0.000138*(1/ms)*1*mV*1.12)/(mV*exprel((v+56*mV)/(1*mV))) : Hz

        m_inf = alpha_m/(alpha_m + beta_m) : 1
        h_inf = alpha_h/(alpha_h + beta_h) : 1
        mp_inf = alpha_mp/(alpha_mp + beta_mp) : 1
        s_inf = alpha_s/(alpha_s + beta_s) : 1

        tau_m = 1/(alpha_m + beta_m) : second
        tau_h = 1/(alpha_h + beta_h) : second
        tau_mp = 1/(alpha_mp + beta_mp) : second
        tau_s = 1/(alpha_s + beta_s) : second

        '''
    const_potentials = {'EL': leak_potential, 'ER': rest_potential, 'ENa': na_potential, 'EK': k_potential}
    myelinated = SpatialNeuron(morphology=my_morpho, model=eqs, Cm=membrane_c, Ri=axial_rho,
                               method='exponential_euler', namespace=const_potentials)
    myelinated.v = 'ER'
    myelinated.main.gNaf, myelinated.main.gNap, myelinated.main.gK, myelinated.main.gL = g_na_f, g_na_p, g_k, g_l_node
    myelinated.p1.gNaf, myelinated.p1.gNap, myelinated.p1.gK, myelinated.p1.gL, myelinated.p1.Cm = (0, 0, 0, g_l_inter,
                                                                                                    para_c)
    myelinated.p1.i1.gNaf, myelinated.p1.i1.gNap, myelinated.p1.i1.gK = 0, 0, 0
    myelinated.p1.i1.gL, myelinated.p1.i1.Cm = g_l_inter, inter_c
    myelinated.p1.i1.p2.gNaf, myelinated.p1.i1.p2.gNap, myelinated.p1.i1.p2.gK = 0, 0, 0
    myelinated.p1.i1.p2.gL, myelinated.p1.i1.p2.Cm = g_l_inter, para_c
    myelinated.p1.i1.p2.n2.gNaf, myelinated.p1.i1.p2.n2.gNap = g_na_f, g_na_p
    myelinated.p1.i1.p2.n2.gK, myelinated.p1.i1.p2.n2.gL = g_k, g_l_node
    myelinated.p1.i1.p2.n2.p3.gNaf, myelinated.p1.i1.p2.n2.p3.gNap, myelinated.p1.i1.p2.n2.p3.gK = 0, 0, 0
    myelinated.p1.i1.p2.n2.p3.gL, myelinated.p1.i1.p2.n2.p3.Cm = g_l_inter, para_c
    myelinated.p1.i1.p2.n2.p3.i2.gNaf, myelinated.p1.i1.p2.n2.p3.i2.gNap, myelinated.p1.i1.p2.n2.p3.i2.gK = 0, 0, 0
    myelinated.p1.i1.p2.n2.p3.i2.gL, myelinated.p1.i1.p2.n2.p3.i2.Cm = g_l_inter, inter_c
    myelinated.p1.i1.p2.n2.p3.i2.p4.gNaf, myelinated.p1.i1.p2.n2.p3.i2.p4.gNap, myelinated.p1.i1.p2.n2.p3.i2.p4.gK = (0,
                                                                                                                      0,
                                                                                                                      0)
    myelinated.p1.i1.p2.n2.p3.i2.p4.gL, myelinated.p1.i1.p2.n2.p3.i2.p4.Cm = g_l_inter, para_c
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.gNaf, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.gNap = g_na_f, g_na_p
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.gK, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.gL = g_k, g_l_node
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.gNaf, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.gNap = 0, 0
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.gK = 0
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.gL, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.Cm = g_l_inter, para_c
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.gNaf, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.gNap = 0, 0
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.gK = 0
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.gL, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.Cm = g_l_inter, inter_c
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.gNaf, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.gNap = 0, 0
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.gK = 0
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.gL, myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.Cm = g_l_inter, para_c
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.n4.gNaf = g_na_f
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.n4.gNap = g_na_p
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.n4.gK = g_k
    myelinated.p1.i1.p2.n2.p3.i2.p4.n3.p5.i3.p6.n4.gL = g_l_node

    elect = PointElectrode(current_amp=-11 * uA, frequency=200 * Hz, rx=1000 * um, ry=1000 * um, rz=500 * um,
                           sigma_ex=0.2 * siemens / meter, origin=6, morphology=my_morpho, node_length=1 * um,
                           internode_length=110 * um, paranode_length=3 * um)

    defaultclock.dt = 0.005 * ms
    elect.i_applied_spatial(node_length, paranode_length, internode_length, node_diam, internode_diam,
                                        axial_rho, my_morpho, myelinated)
    run(10 * ms, report='text')
    assert abs(myelinated.i_appl[0]) == abs(myelinated.i_appl[12])
    assert abs(myelinated.i_appl[5]) == abs(myelinated.i_appl[7])
    assert myelinated.i_appl[4] == myelinated.i_appl[8]



