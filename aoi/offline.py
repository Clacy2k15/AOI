# 推論程序的離線模式：停用套件自動連網與遙測，並限制 Python 的 DNS／TCP／UDP 目的地。
# localhost 仍需開放，因為 Gradio GUI 與原生 Qwen server 透過本機 HTTP 溝通。
# 安裝工具不呼叫此函式，才能在初次 setup 時下載套件與模型；原生程序不受 Python audit hook 攔截。

"""Local-only runtime: cached assets and loopback services, no Internet fallback."""
from __future__ import annotations
import ipaddress
import os
import sys

_enabled = False


# 設定各套件的離線環境變數，並且只安裝一次全程序 audit hook。
# 每次呼叫仍重設環境旗標；_enabled 用來防止同一程序重複註冊網路攔截器。
def enable_offline():
    global _enabled
    for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'GRADIO_ANALYTICS_ENABLED': 'False', 'DO_NOT_TRACK': '1'}.items():
        os.environ[key] = value
    if _enabled:
        return

    # 核對目的主機，只允許 localhost、空綁定位址及 IPv4／IPv6 loopback。
    # 其他主機名在 DNS 解析前就拒絕；IPv6 的 zone suffix 會先去除再判斷位址。
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

    # 攔截 getaddrinfo、connect 與 sendto 三類 Python socket 稽核事件。
    # 不同事件的參數位置不同，因此分別抽出主機欄位再交給 check_host。
    def audit(event, args):
        if event == 'socket.getaddrinfo':
            check_host(args[0])
        elif event == 'socket.connect' and isinstance(args[1], tuple):
            check_host(args[1][0])
        elif event == 'socket.sendto' and isinstance(args[-1], tuple):
            check_host(args[-1][0])

    # audit hook 一旦加入就會持續作用於此 Python 程序，故必須在模型與 GUI 啟動前設定。
    sys.addaudithook(audit)
    _enabled = True
