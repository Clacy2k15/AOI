# 啟動／重用／停止專案原生 llama-server 的命令列工具，Python 只負責程序與設定管理。
# 語言 GGUF 與視覺 mmproj 必須成對使用；服務只監聽 127.0.0.1，埠號取自 qwen.base_url。
# 程序紀錄含 PID、命令與檔案路徑，停止或重用前必須核對真實程序與 /props。

"""Launch the existing native llama-server; no Python model server."""
import argparse
from urllib.parse import urlsplit
import json
import subprocess
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi.common import ROOT, load_config, load_inference, write_json, read_json
from aoi.velm import verify_server_model, stop_recorded_server
from aoi.offline import enable_offline

enable_offline()


# 把已解析的 Qwen 設定轉成 subprocess 參數清單，先檢查執行檔與兩份模型都存在。
# 使用 list 而非 shell 拼接命令，可正確處理含空白路徑；回傳命令本身，不在此啟動程序。
def server_command(settings: dict) -> list[str]:
    options = settings["qwen"]
    executable = ROOT / options["executable"]
    if not executable.is_file():
        matches = list((ROOT / "models/llama.cpp/b11609").rglob("llama-server.exe"))
        if len(matches) == 1:
            executable = matches[0]
        else:
            raise FileNotFoundError("Official Windows llama-server executable missing: " + str(executable))
    for key in ["model_path", "mmproj_path"]:
        if not (ROOT / options[key]).is_file():
            raise FileNotFoundError(ROOT / options[key])
    # 固定關閉 thinking、使用聊天模板並設定 seed；gpu_layers／ctx／batch 等沿用產品設定。
    return [str(executable), "--model", str(ROOT / options["model_path"]), "--mmproj", str(ROOT / options["mmproj_path"]),
            "--fit", "off", "--mmproj-offload", "-lv", "4",
            "--alias", "aoi-qwen", "--host", "127.0.0.1", "--port", str(urlsplit(options["base_url"]).port or 8080), "--gpu-layers", str(options["gpu_layers"]),
            "--ctx-size", str(options["ctx_size"]), "--parallel", str(options["parallel"]), "--threads", str(options["threads"]),
            "--threads-batch", str(options["threads_batch"]), "--batch-size", str(options["batch_size"]),
            "--ubatch-size", str(options["ubatch_size"]), "--image-max-tokens", str(options["image_max_tokens"]),
            "--jinja", "--reasoning", "off", "--chat-template-kwargs", json.dumps({"enable_thinking": False}),
            "--n-predict", str(options["max_tokens"]), "--seed", str(options["seed"])]


# 主入口先解析參數與產品推論設定，再依 dry-run、stop、重用或新建服務分支執行。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--stop", action="store_true", help="Stop only the verified recorded project process")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths/command without launching")
    args = parser.parse_args()
    settings = load_config(ROOT / args.config)
    aoi_config = load_config(ROOT / settings.get("aoi_config", "configs/aoi.yaml"))
    settings = load_inference(aoi_config, args.config)
    command = server_command(settings)
    # 僅列印已解析命令，不啟動模型；可用來檢查路徑及啟動參數是否正確。
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        sys.exit(0)
    # 停止前先核對權重及程序身分，不以 8080 是否有回應就隨意終止其他服務。
    if args.stop:
        # The expected config must match the running pair before touching a PID.
        verify_server_model(settings["qwen"])
        record = read_json(ROOT / "outputs/qwen_server_process.json")
        stop_recorded_server(record)
        print("Stopped verified project Qwen server PID " + str(record["pid"]))
        sys.exit(0)
    # 埠被占用時不直接另開服務；需確認占用者就是本專案預期的權重組合。
    with socket.socket() as probe:
        occupied = probe.connect_ex(("127.0.0.1", urlsplit(settings["qwen"]["base_url"]).port or 8080)) == 0
    if occupied:
        verify_server_model(settings["qwen"])
        print("Existing server verified against GGUF and mmproj; reuse it.")
        sys.exit(0)
    # 背景模式把輸出接到專案日誌，CREATE_NO_WINDOW 避免另外跳出黑色終端視窗。
    if args.background:
        destination = ROOT / "outputs/qwen_server.log"
        with open(destination, "ab") as stream:
            process = subprocess.Popen(command, cwd=Path(command[0]).parent, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        # 保存之後驗證／停止所需的程序身分。日誌採 append，保留先前啟動的診斷資訊。
        write_json(ROOT / "outputs/qwen_server_process.json", {"pid": process.pid, "command": command,
                    "product": aoi_config["project"]["category"], "shared_server": True,
                    "executable": command[0], "log_path": str(destination), "endpoint": settings["qwen"]["base_url"]})
        print(f"Native server launched hidden, PID {process.pid}")
    else:
        process = subprocess.Popen(command, cwd=Path(command[0]).parent)
        write_json(ROOT / "outputs/qwen_server_process.json", {"pid":process.pid,"command":command,
                   "product":aoi_config["project"]["category"],"shared_server":True,"executable":command[0],
                   "endpoint":settings["qwen"]["base_url"]})
        try:
            sys.exit(process.wait())
        # 前景等待模式收到 Ctrl+C 時終止這次建立的子程序，並等待其確實退出。
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
