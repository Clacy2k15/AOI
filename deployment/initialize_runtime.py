# 安裝後的路徑實體化工具：把可攜模板的 @AOI_ROOT@ 替換成這台電腦的專案絕對路徑。
# 只重建部署 JSON 與更新 bundle 的內容校驗值；checkpoint、門檻數值及前處理演算法不重新訓練。
# 需先安裝鎖定套件，因為最後會從本機 PyTorch 複製 llama.cpp 需要的三個 CUDA runtime DLL。

"""Materialize machine-specific metadata from portable templates, never edit weights."""
import hashlib
import json
from pathlib import Path

# 模板保留這個字面字串，不把作者原電腦的使用者名稱與資料夾路徑寫死到公開部署紀錄。
TOKEN = '@AOI_ROOT@'


# 以串流讀取檔案並計算 SHA256，避免為大檔配置同樣大小的額外記憶體。
# 回傳十六進位字串，用於重建後的 bundle 檔案校驗。
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# 將 relative 接到 root 並 resolve，確認真正目的地仍位於專案根目錄內。
# relative_to 失敗即拋出 ValueError，可攔住 ../、外部絕對路徑及指向外部的連結。
def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


# 遞迴走訪 JSON 的字串、清單及字典值，把根目錄佔位符換成本機絕對路徑。
# 數值、布林值與字典鍵原樣保留，因此模型門檻及校驗索引不被任意改動。
def materialize(value, root):
    if isinstance(value, str):
        return value.replace(TOKEN, str(root.resolve()))
    if isinstance(value, list):
        return [materialize(v, root) for v in value]
    if isinstance(value, dict):
        return {k: materialize(v, root) for k, v in value.items()}
    return value


# 依模板清單建立訓練摘要／ROI bundle，再把新 bundle SHA256 寫入對應推論 YAML。
# 接著複製 PyTorch 隨套件提供的 CUDA DLL；重跑時從同一模板重建，避免把舊機器路徑疊加進來。
def initialize(root):
    import yaml
    templates = json.loads((root / 'deployment/path_templates.json').read_text(encoding='utf-8'))
    for relative, value in templates['files'].items():
        destination = inside(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(materialize(value, root), ensure_ascii=False, indent=2), encoding='utf-8')
    # 先寫完 bundle 才計算雜湊；若先雜湊再換路徑，執行時的嚴格校驗會失敗。
    for relative, key in templates['bundle_configs'].items():
        path = inside(root, relative)
        settings = yaml.safe_load(path.read_text(encoding='utf-8'))
        settings[key]['bundle_sha256'] = sha(inside(root, settings[key]['bundle']))
        path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding='utf-8', newline='\n')
    print('Runtime paths initialized for:', root)


if __name__ == '__main__':
    initialize(Path(__file__).resolve().parents[1])
