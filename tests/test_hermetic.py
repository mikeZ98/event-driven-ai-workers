"""The suite is hermetic: pytest-socket blocks every non-Unix socket."""

import socket

import pytest
from pytest_socket import SocketBlockedError


def test_inet_sockets_are_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_unix_socketpair_allowed_for_asyncio() -> None:
    left, right = socket.socketpair(socket.AF_UNIX)
    left.close()
    right.close()
