# AOI：OK／NG 與 mask

## 版本選擇：本專案提供兩個版本

| 版本 | 分支 | 功能 | Release |
|---|---|---|---|
| 完整版（含瑕疵分類） | [main](https://github.com/Clacy2k15/AOI/tree/main) | OK／NG、mask、定位疊圖，以及 VELM／Qwen 瑕疵名稱與分類信心 | [v1.0.1 完整版](https://github.com/Clacy2k15/AOI/releases/tag/v1.0.1) |
| 精簡版（只做偵測） | [detection-only](https://github.com/Clacy2k15/AOI/tree/detection-only) | OK／NG、mask 與定位疊圖；不含瑕疵分類 | [detection-only-v1.0.0 精簡版](https://github.com/Clacy2k15/AOI/releases/tag/detection-only-v1.0.0) |

兩版保留相同的 15 種產品偵測模型與門檻。精簡版不需要 VELM／Qwen、GGUF、mmproj 或 llama.cpp，並使用專屬的偵測模型下載包。請依需求選擇版本，**分別安裝到不同資料夾，避免覆蓋既有版本**。

**本頁以下為精簡版的操作說明。**

`detection-only` 是獨立的精簡分支，支援原本 15 種產品。介面與 CLI 只提供 OK／NG、原圖尺寸二值 mask、定位疊圖及結果下載。

沿用已訓練的 EfficientAD＋PatchCore、逐產品前處理、選定模型及門檻。移除 VELM／Qwen 分類、分類信心、瑕疵名稱、三圖輸入、分類快取與服務啟動工具。推論不需要 GGUF、mmproj 或 llama.cpp。

## 獨立安裝

請從 [精簡版 Release](https://github.com/Clacy2k15/AOI/releases/tag/detection-only-v1.0.0) 下載 `aoi-detection-only-v1.0.0-source.zip` 或 `Source code (zip)`，解壓到**新的資料夾**；原有版本可繼續使用。

需要 Windows x64、Python 3.12（含 `py` launcher）、支援 CUDA 的 NVIDIA GPU 與驅動。已驗證主機為 RTX 4070。初次安裝需要網路，安裝完成後可離線推論。

若已有完整 AOI 版本，指定其根目錄即可複製並核對 30 個偵測 checkpoint。來源程式與模型只讀，不修改來源：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -ModelsFrom 'C:\path\to\existing\AOI'
```

新安裝可直接執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

本版固定使用專屬 `detection-only-v1.0.0` Release，偵測模型包約 **4.03 GB、共 4 段**，只包含 30 個 EfficientAD／PatchCore checkpoint（含 capsule／pill ROI 模型），不含任何 Qwen／llama.cpp 資源。安裝器會自動下載，並核對分段、整包及模型 SHA256。已有模型時使用 `-ModelsFrom` 可避免重複下載。

手動下載時，請將本版四個 `aoi-detection-runtime.zip.part001`～`part004` 放在同一資料夾，再用 `-AssetsDirectory 'D:\assets'`。完全離線安裝另需依 `requirements.e5.lock.txt` 準備 `wheelhouse`，再加上 `-Offline`。

## 圖形介面

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-demo.ps1
```

開啟 `http://127.0.0.1:7860`，選產品後上傳影像，或填入本機影像／資料夾路徑。畫面顯示判決、mask 與疊圖；ZIP 包含全部結果。若與原版同時啟動，請改用另一個埠號：

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --port 7861
```

## 命令列

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --config configs\products\bottle\aoi.yaml --input 'D:\images' --output outputs\inspection
```

`metal_nut` 使用 `configs/aoi.yaml`；其餘產品使用 `configs/products/<product>/aoi.yaml`。支援 PNG、JPEG、BMP、TIFF，資料夾會遞迴處理。

每張保存 `mask.png`、`overlay.png`、`result.json`；整批另存 `results.json`、`results.csv`。JSON 包含產品、來源影像、OK／NG、輸出路徑及錯誤。執行失敗時 `decision` 為 `null`，畫面顯示「執行失敗」，CLI 回傳非零退出碼，不把錯誤判成良品或不良品。

## 判定方式

- 任一偵測器影像分數達到自己的 image 門檻即為 NG；兩者皆未達門檻為 OK。
- 各自分數圖依 pixel 門檻二值化後做 OR，產生原圖尺寸的 mask：白色 255 為候選異常，黑色 0 為背景。
- 影像與像素門檻獨立，因此 NG 可能有空 mask，OK 也可能有局部 mask。保留原有校準行為，不由 mask 是否為空推翻判決。
- capsule／pill 保留已採用的 ROI 模型、回映及完整性檢查。

## 驗證與程式入口

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe deployment\check_runtime.py --full
```

驗證紀錄見 [VALIDATION.md](VALIDATION.md)。主要程式為 `scripts/demo.py`、`scripts/predict.py`、`aoi/pipeline.py`、`aoi/fusion.py`。沒有訓練或分類入口；模型配方欄位保留供權重身分校驗。

MVTec 資料與範例的使用條款見 `third_party/MVTec-AD-LICENSE.txt`。
