# 推論發佈包內的精簡 Qwen 服務管理工具，供 GUI 切換產品時呼叫。
# 只保留原批次工具的 ensure_server，不匯入資料準備或訓練入口。
# 不同產品若共用同一 GGUF／mmproj 可共用服務；切換權重前先核對並停止已記錄程序。

import json, subprocess, sys, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta
from aoi.common import ROOT, load_config, load_inference, read_json
from aoi.velm import recorded_server_command, stop_recorded_server, verify_server_model

# 回傳帶 Asia/Taipei 對應 UTC+8 時差的 ISO 8601 字串。
# 用於服務核對紀錄，避免受執行電腦的預設時區影響。
def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()

# 載入產品設定，檢查既有程序是否使用同一權重，啟動或重用 server 並輪詢 /health。
# 最長等 180 秒，成功後再次核對 /props 與程序命令列，回傳已確認的執行身分。
def ensure_server(config_path):
    raw=load_config(config_path)
    config=load_config(raw.get('aoi_config','configs/aoi.yaml'))
    settings=load_inference(config,config_path,require_calibrated=True)
    record_file=ROOT/'outputs/qwen_server_process.json'
    if record_file.exists():
        record=read_json(record_file)
        try: recorded_server_command(record)
        # 只接受「已記錄程序已結束」這種情況；PID 被重用或權重不符等身分問題不能直接忽略。
        except RuntimeError as e:
            if 'not running' not in str(e): raise
        else:
            different=any(Path(record['command'][record['command'].index(flag)+1]).resolve()!=(ROOT/settings['qwen'][key]).resolve() for key,flag in [('model_path','--model'),('mmproj_path','--mmproj')])
            if different: stop_recorded_server(record)
    # 沿用目前 Python 直譯器與專案根目錄啟動工具，避免誤用系統上的另一個套件環境。
    subprocess.run([sys.executable,str(ROOT/'scripts/start_qwen.py'),'--config',str(config_path),'--background'],cwd=ROOT,check=True)
    # 以 monotonic 計時避免系統時鐘調整干擾，輪詢間隔 1 秒，逾時就明確回報失敗。
    deadline=time.monotonic()+180
    while True:
        try:
            with urllib.request.urlopen(settings['qwen']['base_url']+'/health',timeout=5) as r:
                if json.load(r).get('status')=='ok': break
        except Exception:
            if time.monotonic()>deadline: raise RuntimeError('Native Qwen did not become ready within 180 seconds')
            time.sleep(1)
    verify_server_model(settings['qwen'])
    with urllib.request.urlopen(settings['qwen']['base_url']+'/props',timeout=10) as r: props=json.load(r)
    record=read_json(record_file)
    return {'checked_at':now(),'pid':record['pid'],'actual_command':recorded_server_command(record),
            'GGUF':settings['qwen']['model_path'],'mmproj':settings['qwen']['mmproj_path'],
            'server_model_path':props['model_path'],'pair_verified':True,'properties':props}
