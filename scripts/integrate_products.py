import json, subprocess, sys, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta
from aoi.common import ROOT, load_config, load_inference, read_json
from aoi.velm import recorded_server_command, stop_recorded_server, verify_server_model

def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()

def ensure_server(config_path):
    raw=load_config(config_path)
    config=load_config(raw.get('aoi_config','configs/aoi.yaml'))
    settings=load_inference(config,config_path,require_calibrated=True)
    record_file=ROOT/'outputs/qwen_server_process.json'
    if record_file.exists():
        record=read_json(record_file)
        try: recorded_server_command(record)
        except RuntimeError as e:
            if 'not running' not in str(e): raise
        else:
            different=any(Path(record['command'][record['command'].index(flag)+1]).resolve()!=(ROOT/settings['qwen'][key]).resolve() for key,flag in [('model_path','--model'),('mmproj_path','--mmproj')])
            if different: stop_recorded_server(record)
    subprocess.run([sys.executable,str(ROOT/'scripts/start_qwen.py'),'--config',str(config_path),'--background'],cwd=ROOT,check=True)
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
