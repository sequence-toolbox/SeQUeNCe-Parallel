import numpy as np

from sequence.components.beam_splitter import BeamSplitter, FockBeamSplitter2
from sequence.components.detector import FockDetector
from sequence.components.photon import Photon
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization
from sequence.topology.node import Node

np.random.seed(0)
SEED = 0


def test_BeamSplitter_init():
    tl = Timeline()
    bs = BeamSplitter("bs", tl)
    bs.add_receiver(None)
    bs.add_receiver(None)
    tl.init()


class Owner:
    def __init__(self):
        self.generator = np.random.default_rng(SEED)

    def get_generator(self):
        return self.generator


class Receiver:
    def __init__(self, tl):
        self.timeline = tl
        self.log = []

    def get(self, photon=None):
        self.log.append((self.timeline.now()))

    def reset(self):
        self.log = []


def test_BeamSplitter_get():
    tl = Timeline()
    bs = BeamSplitter("bs", tl)
    own = Owner()
    bs.owner = own
    receiver0 = Receiver(tl)
    bs.add_receiver(receiver0)
    receiver1 = Receiver(tl)
    bs.add_receiver(receiver1)

    frequency = 8e7
    start_time = 0
    basis_len = 1000
    basis_list = []

    # z-basis states, measurement
    for i in range(basis_len):
        basis_list.append(0)

    bs.set_basis_list(basis_list, start_time, frequency)

    bits = []
    for i in range(basis_len):
        time = 1e12 / frequency * i
        tl.time = time
        bit = np.random.randint(2)
        bits.append(bit)
        photon = Photon(str(i), tl, quantum_state=polarization["bases"][0][bit])
        bs.get(photon)

    for i in range(basis_len):
        time = 1e12 / frequency * i
        r_i = bits[i]
        assert time in bs._receivers[r_i].log

    # x-basis states, measurement
    receiver0.log = []
    receiver1.log = []
    basis_list = []
    for i in range(basis_len):
        basis_list.append(1)

    bs.set_basis_list(basis_list, start_time, frequency)

    bits2 = []
    for i in range(basis_len):
        time = 1e12 / frequency * i
        tl.time = time
        bit = np.random.randint(2)
        bits2.append(bit)
        photon = Photon(str(i), tl, quantum_state=polarization["bases"][1][bit])
        bs.get(photon)

    for i in range(basis_len):
        time = 1e12 / frequency * i
        r_i = bits2[i]
        assert time in bs._receivers[r_i].log

    # z-basis states, x-basis measurement
    receiver0.log = []
    receiver1.log = []
    basis_list = []
    for i in range(basis_len):
        basis_list.append(1)

    bs.set_basis_list(basis_list, start_time, frequency)

    bits = []
    for i in range(basis_len):
        time = 1e12 / frequency * i
        tl.time = time
        bit = np.random.randint(2)
        bits.append(bit)
        photon = Photon(str(i), tl, quantum_state=polarization["bases"][0][bit])
        bs.get(photon)

    print(len(receiver1.log), len(receiver0.log))
    true_counter, false_counter = 0, 0
    for i in range(basis_len):
        time = 1e12 / frequency * i
        r_i = bits[i]
        if time in bs._receivers[r_i].log:
            true_counter += 1
        else:
            false_counter += 1
    assert true_counter / basis_len - 0.5 < 0.1


# def test_FockBeamSplitter_get():
#     NUM_TRIALS = 1000
#     psi_minus = [complex(0), complex(sqrt(1 / 2)), -complex(sqrt(1 / 2)), complex(0)]
#
#     tl = Timeline(formalism="density_matrix")
#     rec_0 = Receiver(tl)
#     rec_1 = Receiver(tl)
#
#     bs = FockBeamSplitter("bs", tl)
#     bs.add_receiver(rec_0)
#     bs.add_receiver(rec_1)
#
#     tl.init()
#
#     # measure unentangled
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert abs(len(rec_0.log) / len(rec_1.log)) - 1 < 0.1
#
#     # measure entangled, no phase
#     tl.time = 0
#     rec_0.reset()
#     rec_1.reset()
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         p0.entangle(p1)
#         p0.set_state(psi_minus)
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert abs(len(rec_0.log) / len(rec_1.log)) - 1 < 0.1
#
#     # measure entangled, pi/2 phase
#     tl.time = 0
#     rec_0.reset()
#     rec_1.reset()
#     circuit = Circuit(1)
#     circuit.phase(0, pi/2)
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         p0.entangle(p1)
#         p0.set_state(psi_minus)
#         tl.quantum_manager.run_circuit(circuit, [p0.quantum_state])
#
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert len(rec_1.log) == NUM_TRIALS
#
#     # measure entangled, 3pi/2 phase
#     tl.time = 0
#     rec_0.reset()
#     rec_1.reset()
#     circuit = Circuit(1)
#     circuit.phase(0, 3*pi/2)
#     for _ in range(NUM_TRIALS):
#         p0 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p1 = Photon("", tl, encoding_type=absorptive, use_qm=True)
#         p0.is_null = True
#         p0.entangle(p1)
#         p0.set_state(psi_minus)
#         tl.quantum_manager.run_circuit(circuit, [p0.quantum_state])
#
#         bs.get(p0)
#         bs.get(p1)
#         tl.time += 1
#
#     assert len(rec_0.log) == NUM_TRIALS


def test_FockBeamSplitter2():
    """ Quantum transduction via entanglemenet swapping (node2).
        A fock beamsplitter is attached to two detectors
    """
    tl = Timeline()
    node1 = Node('Node1', tl)
    node2 = Node('Node2', tl)  # swapping node
    node3 = Node('Node3', tl)
    src_list = [node1, node3]

    detector1 = FockDetector(node2.name + '.detector1', tl, efficiency=0.5)
    detector2 = FockDetector(node2.name + '.detector2', tl, efficiency=0.5)

    fockbeamsplitter2 = FockBeamSplitter2('fock_beamsplitter2', node2, tl, efficiency=0.9, photon_counter=0, src_list=src_list)
    fockbeamsplitter2.add_outputs([detector1, detector2])

    photon = Photon('photon', tl)

    # receives 10 photons from node1 and 10 photons from node3
    photon_number = 10
    for _ in range(photon_number):
        fockbeamsplitter2.get(node1.name, photon)
    for _ in range(photon_number):
        fockbeamsplitter2.get(node2.name, photon)

    assert fockbeamsplitter2.photon_counter == photon_number * 2
