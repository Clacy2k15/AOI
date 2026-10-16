"""Local-only runtime: cached assets and loopback services, no Internet fallback."""
from __future__ import annotations
import ipaddress
import os
import sys

_enabled = False


def enable_offline():
    global _enabled
    for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'GRADIO_ANALYTICS_ENABLED': 'False', 'DO_NOT_TRACK': '1'}.items():
        os.environ[key] = value
    if _enabled:
        return

    def check_host(host):
        if isinstance(host, bytes):
            host = host.decode('ascii')
        if host in (None, '', 'localhost'):
            return
        try:
            if ipaddress.ip_address(host.split('%')[0]).is_loopback:
                return
        except ValueError:
            pass
        raise PermissionError('Offline AOI runtime blocks external network access: ' + str(host))

    def audit(event, args):
        if event == 'socket.getaddrinfo':
            check_host(args[0])
        elif event == 'socket.connect' and isinstance(args[1], tuple):
            check_host(args[1][0])
        elif event == 'socket.sendto' and isinstance(args[-1], tuple):
            check_host(args[-1][0])

    sys.addaudithook(audit)
    _enabled = True
