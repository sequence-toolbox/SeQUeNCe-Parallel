"""Models for simulation of optical fiber channels.

This module introduces the abstract OpticalChannel class for general optical fibers.
It also defines the QuantumChannel class for transmission of qubits/photons and the ClassicalChannel class for transmission of classical control messages.
OpticalChannels must be attached to nodes on both ends.
"""

import heapq as hq
import gmpy2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..topology.node import Node
    from ..components.photon import Photon
    from ..message import Message

from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils import log
from ..constants import SPEED_OF_LIGHT, MICROSECOND, SECOND, EPSILON

gmpy2.get_context().precision = 200
EPSILON_MPFR = gmpy2.mpfr(EPSILON)
PS_PER_SECOND = gmpy2.mpz(SECOND)


class OpticalChannel(Entity):
    """Parent class for optical fibers.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (str): name of the node at receiving end of optical channel.
        attenuation (float): attenuation of the fiber (in dB/m).
        distance (float): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
    """

    def __init__(self, name: str, timeline: "Timeline", attenuation: float, distance: float,
                 polarization_fidelity: float, light_speed: float):
        """Constructor for abstract Optical Channel class.

        Args:
            name (str): name of the beamsplitter instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/m).
            distance (float): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
            light_speed (float): speed of light within the fiber (in m/ps).
        """
        log.logger.info("Create channel {}".format(name))

        Entity.__init__(self, name, timeline)
        self.sender = None
        self.receiver = None
        self.attenuation = attenuation
        self.distance = distance  # (measured in m)
        self.polarization_fidelity = polarization_fidelity
        self.light_speed = light_speed  # used for photon timing calculations (measured in m/ps)

    def init(self) -> None:
        pass

    def set_distance(self, distance: float) -> None:
        self.distance = distance


class QuantumChannel(OpticalChannel):
    """Optical channel for transmission of photons/qubits.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (str): name of the node at receiving end of optical channel.
        attenuation (float): attenuation of the fiber (in dB/m).
        distance (float): length of the fiber (in m).
        polarization_fidelity (float): probability of no polarization error for a transmitted qubit.
        light_speed (float): speed of light within the fiber (in m/ps).
        loss (float): loss rate for transmitted photons (determined by attenuation).
        delay (int): delay (in ps) of photon transmission (determined by light speed, distance).
        frequency (float): maximum frequency of qubit transmission (in Hz).
    """

    def __init__(self, name: str, timeline: "Timeline", attenuation: float, distance: float,
                 polarization_fidelity: float = 1.0, light_speed: float = SPEED_OF_LIGHT, frequency: float = 8e7):
        """Constructor for Quantum Channel class.

        Args:
            name (str): name of the quantum channel instance.
            timeline (Timeline): simulation timeline.
            attenuation (float): loss rate of optical fiber (in dB/m).
            distance (float): length of fiber (in m).
            polarization_fidelity (float): probability of no polarization error for a transmitted qubit (default 1).
            light_speed (float): speed of light within the fiber (in m/ps).
            delay (int): delay (in ps) of photon transmission (determined by light speed, distance).
            loss (float): loss rate for transmitted photons (determined by attenuation).
            frequency (float): maximum frequency of qubit transmission (in Hz) (default 8e7).
        """

        super().__init__(name, timeline, attenuation, distance, polarization_fidelity, light_speed)
        self.delay: int = -1
        self.loss: float = 1
        self.frequency: float = frequency  # maximum frequency for sending qubits (measured in Hz)
        self.send_bins: list = []

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        self.delay = round(self.distance / self.light_speed)
        self.loss = 1 - 10 ** (self.distance * self.attenuation / -10)

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the quantum channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending qubits.
            receiver (str): name of node receiving qubits.
        """

        log.logger.info("Set {}, {} as ends of quantum channel {}".format(sender.name, receiver, self.name))
        self.sender = sender
        self.receiver = receiver
        sender.assign_qchannel(self, receiver)

    def transmit(self, qubit: "Photon", source: "Node") -> None:
        """Method to transmit photon-encoded qubits.

        Args:
            qubit (Photon): photon to be transmitted.
            source (Node): source node sending the qubit.

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        log.logger.info("{} send qubit with state {} to {} by Channel {}".format(
                        self.sender.name, qubit.quantum_state, self.receiver, self.name))

        assert self.delay >= 0 and self.loss < 1, "QuantumChannel init() function has not been run for {}".format(self.name)
        assert source == self.sender

        # remove lowest time bin
        if len(self.send_bins) > 0:
            time = -1
            while time < self.timeline.now():
                time_bin = hq.heappop(self.send_bins)
                time = self.timebin_to_time(time_bin, self.frequency)
            assert time == self.timeline.now(), "qc {} transmit method called at invalid time".format(self.name)

        # check if photon state using Fock representation
        if qubit.encoding_type["name"] == "fock":
            key = qubit.quantum_state  # if using Fock representation, the `quantum_state` field is the state key.
            # apply loss channel on photonic statex
            self.timeline.quantum_manager.add_loss(key, self.loss)

            # schedule receiving node to receive photon at future time determined by light speed
            future_time = self.timeline.now() + self.delay
            process = Process(self.receiver, "receive_qubit", [source.name, qubit])
            event = Event(future_time, process)
            self.timeline.schedule(event)

        # if not using Fock representation, check if photon kept
        elif (self.sender.get_generator().random() > self.loss) or qubit.is_null:
            if self._receiver_on_other_tl():
                self.timeline.quantum_manager.move_manage_to_server(qubit.quantum_state)

            if qubit.is_null:
                qubit.add_loss(self.loss)

            # check if polarization encoding and apply necessary noise
            if qubit.encoding_type["name"] == "polarization" and self.sender.get_generator().random() > self.polarization_fidelity:
                qubit.random_noise(self.get_generator())

            # schedule receiving node to receive photon at future time determined by light speed
            future_time = self.timeline.now() + self.delay
            process = Process(self.receiver, "receive_qubit", [source.name, qubit])
            event = Event(future_time, process)
            self.timeline.schedule(event)

        # if not using Fock representation, if photon lost, exit
        else:
            pass

    def time_to_timebin(self, time: int, frequency: float) -> int:
        """Convert simulation time to time bin.
           Use the gmpy2.mpfr for high precision floating points.
           The precision is set to 200 bits, equivalent to around 54 significant decimal digits.
           The float in Python is 64 bits,   equivalent to around 16 significant decimal digits.

        Args:
            time (int): simulation time (picoseconds) to convert.
            frequency (float): frequency of the channel.
        Returns:
            int: time bin corresponding to the given simulation time.
        """
        time = gmpy2.mpfr(time)
        frequency = gmpy2.mpfr(frequency)
        time_bin = time * frequency / PS_PER_SECOND
        if time_bin - gmpy2.floor(time_bin) > EPSILON_MPFR:
            time_bin = int(time_bin) + 1       # round to the next time bin
        else:
            time_bin = int(time_bin)
        return time_bin

    def timebin_to_time(self, time_bin: int, frequency: float) -> int:
        """Convert time bin to simulation time (picoseconds).
           Use the gmpy2.mpz  for high precision integers.
           Use the gmpy2.mpfr for high precision floating points.

        Args:
            time_bin (int): time bin to convert.
            frequency (float): frequency of the channel.

        Returns:
            int: simulation time (picoseconds) corresponding to the given time bin.
        """
        time_bin = gmpy2.mpz(time_bin)
        frequency = gmpy2.mpfr(frequency)
        time = gmpy2.mpfr(time_bin * PS_PER_SECOND) / frequency
        return int(time)

    def schedule_transmit(self, min_time: int) -> int:
        """Method to schedule a time for photon transmission.

        Quantum Channels are limited by a frequency of transmission.
        This method returns the next available time for transmitting a photon.
        
        Args:
            min_time (int): minimum simulation time for transmission.

        Returns:
            int: simulation time for next available transmission window.
        """
        min_time = max(min_time, self.timeline.now())
        time_bin = self.time_to_timebin(min_time, self.frequency)

        # find earliest available time bin
        while time_bin in self.send_bins:
            time_bin += 1
        hq.heappush(self.send_bins, time_bin)

        time = self.timebin_to_time(time_bin, self.frequency)
        return time

    def _receiver_on_other_tl(self) -> bool:
        return self.timeline.get_entity_by_name(self.receiver) is None


class ClassicalChannel(OpticalChannel):
    """Optical channel for transmission of classical messages.

    Classical message transmission is assumed to be lossless.

    Attributes:
        name (str): label for channel instance.
        timeline (Timeline): timeline for simulation.
        sender (Node): node at sending end of optical channel.
        receiver (str): name of the node at receiving end of optical channel.
        distance (float): length of the fiber (in m).
        delay (float): delay (in ps) of message transmission (default distance / light_speed).
    """

    def __init__(self, name: str, timeline: "Timeline", distance: float, delay: int = -1):
        """Constructor for Classical Channel class.

        Args:
            name (str): name of the classical channel instance.
            timeline (Timeline): simulation timeline.
            distance (float): length of the fiber (in m).
            delay (float): delay (in ps) of message transmission (default distance / light_speed).
        """

        super().__init__(name, timeline, 0, distance, 0, SPEED_OF_LIGHT)
        if delay == -1:
            self.delay = round(distance / self.light_speed + 10 * MICROSECOND)
        else:
            self.delay = round(delay)

    def set_ends(self, sender: "Node", receiver: str) -> None:
        """Method to set endpoints for the classical channel.

        This must be performed before transmission.

        Args:
            sender (Node): node sending classical messages.
            receiver (str): name of node receiving classical messages.
        """

        log.logger.info("Set {}, {} as ends of classical channel {}".format(sender.name, receiver, self.name))
        self.sender = sender
        self.receiver = receiver
        sender.assign_cchannel(self, receiver)

    def transmit(self, message: "Message", source: "Node", priority: int) -> None:
        """Method to transmit classical messages.

        Args:
            message (Message): message to be transmitted.
            source (Node): node sending the message.
            priority (int): priority of transmitted message (to resolve message reception conflicts).

        Side Effects:
            Receiver node may receive the qubit (via the `receive_qubit` method).
        """

        log.logger.info("{} send message {} to {} by Channel {}".format(self.sender.name, message, self.receiver, self.name))
        assert source == self.sender

        future_time = round(self.timeline.now() + int(self.delay))
        process = Process(self.receiver, "receive_message", [source.name, message])
        event = Event(future_time, process, priority)
        self.timeline.schedule(event)
