# AOI — 15 種產品離線瑕疵檢測

## 版本選擇：本專案提供兩個版本

| 版本 | 分支 | 功能 | Release |
|---|---|---|---|
| 完整版（含瑕疵分類） | [main](https://github.com/Clacy2k15/AOI/tree/main) | OK／NG、mask、定位疊圖，以及 VELM／Qwen 瑕疵名稱與分類信心 | [v1.0.1 完整版](https://github.com/Clacy2k15/AOI/releases/tag/v1.0.1) |
| 精簡版（只做偵測） | [detection-only](https://github.com/Clacy2k15/AOI/tree/detection-only) | OK／NG、mask 與定位疊圖；不含瑕疵分類 | [detection-only-v1.0.0 精簡版](https://github.com/Clacy2k15/AOI/releases/tag/detection-only-v1.0.0) |

兩版保留相同的 15 種產品偵測模型與門檻。精簡版不需要 VELM／Qwen、GGUF、mmproj 或 llama.cpp，並使用專屬的偵測模型下載包。請依需求選擇版本，**分別安裝到不同資料夾，避免覆蓋既有版本**。

**本頁以下為完整版的操作說明。**

使用已訓練的 EfficientAD、PatchCore 與微調 Qwen3.5-4B，在本機對新圖片判定 OK／NG、定位疑似瑕疵及分類。15 類產品需由使用者選擇；新圖片應具有與訓練資料相近的拍攝方式、尺寸和背景。

v1.0.1 為全部 24 個 Python 檔案加入詳細中文註釋，涵蓋 134 個類別／函式／方法及重要演算法區塊；模型資源沿用 v1.0.0。

## 下載與啟動（Windows）

1. 安裝 **Python 3.12 x64**，保留 Windows `py` launcher，並安裝適用於你的 NVIDIA 顯示卡驅動。此版本使用 CUDA，尚未提供 CPU／macOS 版本。原開發機為 RTX 4070；其他顯示卡與顯存需求尚未逐一驗證。
2. 在 GitHub 選擇 **Code → Download ZIP** 並解壓，或 clone 本倉庫。
3. 在專案資料夾開啟 PowerShell，執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-demo.ps1
```

第一次安裝需要網路，會下載固定版本的 Python 套件及 GitHub Release 模型分段，核對 SHA256、解壓並自動設定本機路徑。模型分段總量以 `release-manifest.json` 為準；建議預留至少 35 GB 空間供模型、下載暫存與套件使用。

開啟 http://127.0.0.1:7860，選產品後上傳圖片或填入圖片／資料夾路徑。安裝完成後，日常推論可斷網執行，不需要 API key。Qwen 服務由 GUI 按需啟動，只監聽 localhost。

## 手動下載／完全離線安裝

若自動下載失敗，可從 [v1.0.0 模型 Release](https://github.com/Clacy2k15/AOI/releases/tag/v1.0.0) 下載**所有** `aoi-runtime.zip.partNNN` 到同一資料夾，再執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -AssetsDirectory 'D:\AOI-release-assets'
```

完全離線的新電腦還需要 Python 3.12 安裝程式、驅動及相同 Windows x64／Python 3.12 平台的 wheelhouse。可先在有網路的相同平台準備：

```powershell
py -3.12 -m pip wheel --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.e5.lock.txt --wheel-dir wheelhouse
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Offline -AssetsDirectory 'D:\AOI-release-assets'
```

wheelhouse 不包含於本次模型 Release；完全離線安裝尚待在另一台乾淨電腦驗證。模型下載完成後可保留或自行移除 `.downloads` 中的分段暫存。若搬動已安裝的資料夾，應重建 `.venv` 並重新執行 setup；Python venv 不保證可搬移。

## 命令列推論

```powershell
.\.venv\Scripts\python.exe scripts\start_qwen.py --config configs\products\bottle\inference.yaml --background
.\.venv\Scripts\python.exe scripts\predict.py --config configs\products\bottle\aoi.yaml --input 'D:\images\sample.png'
```

metal_nut 使用 `configs/aoi.yaml`／`configs/inference.yaml`；其他產品使用 `configs/products/<product>/`。輸出包括疊圖、遮罩、JSON、CSV。只要偵測與定位，可加 `--detectors-only` 並略過 Qwen 服務啟動。GUI 與 CLI 共用相同 pipeline。

## 程式碼閱讀順序

建議從 [GUI 入口](scripts/demo.py) 或 [CLI 入口](scripts/predict.py) 開始，再看 [完整 pipeline](aoi/pipeline.py)。所有 Python 檔案均直接附上中文註釋，包含輸入／輸出、重要變數、計算原理及錯誤處理。

| 內容 | 對應原始碼 |
|---|---|
| 設定繼承、檔案路徑與成果身分 | `aoi/common.py` |
| 一般模型載入、RGB／張量與前處理 | `aoi/inference.py`、`aoi/data.py`、`aoi/detectors.py`、`aoi/preprocessing.py` |
| 雙模型 OR、原圖區域與 Qwen 紅圈 | `aoi/fusion.py`、`aoi/capsule_decision.py` |
| capsule／pill 專屬 ROI 與部署校驗 | `aoi/capsule_roi.py`、`aoi/capsule_runtime.py`、`aoi/pill_roi.py`、`aoi/pill_runtime.py` |
| 三圖提示、分類回應與 Qwen 程序管理 | `aoi/velm.py`、`scripts/start_qwen.py`、`scripts/integrate_products.py` |
| 開發指標、PRO／AU-PRO 與像素門檻 | `aoi/evaluation.py`、`aoi/pixel_calibration.py` |
| 離線限制、下載、路徑初始化與啟動自檢 | `aoi/offline.py`、`deployment/` |

已安裝 v1.0.0 的使用者若只更新原始碼，請執行下列指令同步註釋版的前處理程式校驗值；不需重新訓練：

```powershell
.\.venv\Scripts\python.exe deployment\initialize_runtime.py
.\.venv\Scripts\python.exe deployment\check_runtime.py
```

## 發佈內容與限制

- 程式碼放在 Git；大型模型透過固定版本的 GitHub Release 分段提供，不使用 Git LFS。
- 僅包含各產品目前採用的 checkpoint、Qwen Q6_K／mmproj、llama.cpp b11609、15 張正常參考圖及執行所需設定。capsule／pill 包含正式採用的 ROI 模型與校正 bundle。
- 不包含完整 MVTec 訓練／測試影像、Imagenette、訓練中間權重、微調環境、雲端設定或歷史推論紀錄。本發佈用途為新圖片推論；不能直接用它重做整份測試報告。
- 原始權重保持不變；安裝程式只將部署 JSON 的路徑換成本機位置並更新 capsule／pill bundle 的檔案校驗值，不重新訓練或校正門檻。

## 資料與第三方來源

15 張正常參考圖來自 [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，未修改，依 **CC BY-NC-SA 4.0** 提供，授權全文位於 `third_party/MVTec-AD-LICENSE.txt`；官方明確限制資料的商業用途。

Qwen 基底為 [unsloth/Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B)，上游 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 使用 Apache-2.0；本專案以視覺 LoRA 微調後合併並量化為 Q6_K，mmproj 配對匯出。llama.cpp 使用 MIT；相關文件位於 `third_party/`。其他套件保留各自授權。本倉庫未另行替作者選定原創程式碼與自訓權重的新授權。

定位與提示參考 [VELM](https://github.com/Sassanmtr/VELM) 的顯式瑕疵描述與三圖輸入概念；實際提示與類別規則以產品設定為準。
