import numpy as np
from brian2 import defaultclock
from brian2.units.allunits import meter, umeter
from scipy.signal import square


class PointElectrode:
    def __init__(self, current_amp, rx, ry, rz, sigma_ex, origin=None, morphology=None, frequency=None,
                 pulse_width=None, sine_wave=True, duty_cycle=None, node_length=None, internode_length=None,
                 paranode_length=None):
        self.v_applied, self.elect_dist, self.apply_mem_voltage = None, None, None
        if morphology is None:
            self.dictionary_of_lengths = None
        else:
            self.dictionary_of_lengths = {}
            if node_length is None or internode_length is None or paranode_length is None:
                raise TypeError("Please initialize a value for node length, paranode length, and internode length")
            for j in np.arange(0, morphology.total_compartments, 1):
                if j % 4 == 0:
                    self.dictionary_of_lengths[j] = node_length
                elif j % 4 == 1 and j % 2 == 1:
                    self.dictionary_of_lengths[j] = paranode_length
                elif j % 4 != 1 and j % 2 == 1:
                    self.dictionary_of_lengths[j] = paranode_length
                else:
                    self.dictionary_of_lengths[j] = internode_length
        self.rx, self.ry, self.rz, self.origin, = rx, ry, rz, origin
        self.current_amp, self.sigma_ex, self.sine_wave = current_amp, sigma_ex, sine_wave
        if sigma_ex <= 0:
            raise ValueError("Conductivity values should be greater than zero")
        if self.sine_wave:
            self.frequency, self.pw, self.duty_cycle = frequency, None, None
            if frequency <= 0:
                raise ValueError("Frequency should be greater than zero")
        else:
            self.frequency, self.pw, self.duty_cycle = None, pulse_width, duty_cycle
            if pulse_width <= 0:
                raise ValueError("The pulse width should be greater than zero")

    def elect_mem_dist(self):
        self.elect_dist = np.sqrt((self.rx ** 2) + (self.ry ** 2) + (self.rz ** 2))
        return self.elect_dist

    def v_waveform(self):
        self.elect_mem_dist()
        if self.duty_cycle is None:
            self.apply_mem_voltage = ((self.current_amp * np.sin(2 * np.pi * self.frequency * defaultclock.dt)) /
                                      (4 * np.pi * self.elect_dist * self.sigma_ex))
        else:
            self.apply_mem_voltage = ((self.current_amp * square(2 * np.pi * ((0.5 * defaultclock.dt) / self.pw),
                                                                 self.duty_cycle)) / (
                                              4 * np.pi * self.elect_dist * self.sigma_ex))
        return self.apply_mem_voltage

    def origin_to_mid(self, change, target_comp, target_len, morphology):
        k, or_distance = self.origin, 0 * umeter
        while 1 <= k < morphology.total_compartments - 1:
            or_distance = or_distance + 0.5 * (self.dictionary_of_lengths[k] + self.dictionary_of_lengths[k + change])
            k = k + change
            if self.dictionary_of_lengths[k] == target_len and k == target_comp:
                break
        self.ry = or_distance
        return self.ry

    def v_morph(self, node_length, paranode_length, internode_length, end, change, morphology):
        v_applied = {}
        self.ry = 0 * meter
        v_applied[self.origin] = self.v_waveform()
        for j in np.arange(self.origin + change, end, change):
            if j % 4 == 0:
                self.ry = self.origin_to_mid(change, j, node_length, morphology)
                v_applied[j] = self.v_waveform()
            elif j % 4 == 1 and j % 2 == 1:
                self.ry = self.origin_to_mid(change, j, paranode_length, morphology)
                v_applied[j] = self.v_waveform()
            elif j % 4 != 1 and j % 2 == 1:
                self.ry = self.origin_to_mid(change, j, paranode_length, morphology)
                v_applied[j] = self.v_waveform()
            else:
                self.ry = self.origin_to_mid(change, j, internode_length, morphology)
                v_applied[j] = self.v_waveform()
        return v_applied

    def v_applied_spatial(self, node_length, paranode_length, internode_length, morphology):
        self.v_applied = {}
        if self.origin < morphology.total_compartments:
            self.v_applied = self.v_morph(node_length, paranode_length, internode_length, -1, -1, morphology)
            v_applied_compartment1 = self.v_morph(node_length, paranode_length, internode_length,
                                                  morphology.total_compartments, 1, morphology)
            self.v_applied.update(v_applied_compartment1)
        elif self.origin == 0:
            self.v_applied = self.v_morph(node_length, paranode_length, internode_length, morphology.total_compartments,
                                          1, morphology)
        else:
            self.v_applied = self.v_morph(node_length, paranode_length, internode_length, -1, -1, morphology)
        return self.v_applied

    def i_applied_spatial(self, node_length, paranode_length, internode_length, node_diam, internode_diam, axial_rho,
                          morphology, spatial_neuron):
        self.v_applied = self.v_applied_spatial(node_length, paranode_length, internode_length, morphology)
        g_node = ((np.pi / 4) * (node_diam ** 2)) / (node_length * axial_rho)
        g_internode = ((np.pi / 4) * (internode_diam ** 2)) / (internode_length * axial_rho)
        for j in np.arange(0, morphology.total_compartments, 1):
            if j % 4 == 0:
                if j == 0:
                    spatial_neuron[j].i_appl = g_node * (-2 * self.v_applied[j] + self.v_applied[j + 1])
                elif j == morphology.total_compartments - 1:
                    spatial_neuron[j].i_appl = g_node * (self.v_applied[j - 1] - 2 * self.v_applied[j])
                else:
                    spatial_neuron[j].i_appl = g_node * (self.v_applied[j - 1] - 2 * self.v_applied[j] +
                                                         self.v_applied[j + 1])
            # Right paranode
            elif j % 4 == 1 and j % 2 == 1:
                spatial_neuron[j].i_appl = (g_node * (self.v_applied[j - 1] - self.v_applied[j]) +
                                            g_internode * (self.v_applied[j + 1] - self.v_applied[j]))
            # Left paranode
            elif j % 4 != 1 and j % 2 == 1:
                spatial_neuron[j].i_appl = (g_internode * (self.v_applied[j - 1] - self.v_applied[j]) +
                                            g_node * (self.v_applied[j + 1] - self.v_applied[j]))
            else:
                spatial_neuron[j].i_appl = g_internode * (self.v_applied[j - 1] - 2 * self.v_applied[j])
