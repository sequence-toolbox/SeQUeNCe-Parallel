"""Definition of abstract message type."""

from enum import Enum
from abc import ABC


class Message(ABC):
    """Abstract message type inherited by protocol messages."""

    __slots__ = ['msg_type', 'receiver', 'protocol_type', 'payload']

    def __init__(self, msg_type: Enum, receiver: str):
        self.msg_type = msg_type
        self.receiver = receiver
        self.protocol_type = None
        self.payload = None
