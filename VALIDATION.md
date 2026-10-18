# detection-only 分支驗證

2026-10-18（台北時間），從 `main` 的 `420a7e4` 建立獨立 checkout 與 `detection-only` 分支。所有修改、安裝路徑與輸出皆位於新 checkout，保留原版本。

- 13 項回歸測試通過：影像 OR 與門檻等號、像素聯集、NG 空 mask、原圖座標、精簡輸出、偵測／寫檔失敗，以及選擇性解壓、損壞模型保護、路徑越界與來源只讀。
- 15 種產品設定及 CUDA 執行環境檢查通過；模型需求清單只有 30 個偵測 checkpoint。
- 偵測器配方、模型選組、影像／像素門檻、融合規則及正常參考均與來源 `main` 相同。
- 使用 RTX 4070 與既有 Python 3.12 環境驗證；未重新訓練、校準或執行分類。

- 15 種產品各抽取原版已保存的 OK、NG 結果各一張，共 30 張，重新執行真實偵測；所有判決相同，保存的 mask 逐像素完全一致。
- Gradio 可在臨時 localhost 埠啟動並以 HTTP 讀取設定；元件含 mask，無分類／信心／類型／耗時欄位。GUI runner 實際處理 bottle NG，成功產生疊圖、mask、JSON、CSV 與 ZIP，測試服務已關閉。
- 推論與 GUI 過程未匯入 `aoi.velm`。無啟動、停止或變更既有版本的服務。
- 原專案 184 個程式、設定及切分檔案 SHA256 與修改前相同。測試模型透過核對雜湊後複製到獨立 checkout，來源模型只讀。

真實測試的本機詳細紀錄保存在獨立 checkout 的 `outputs/branch-validation/report.json`。本紀錄只驗證部署與既有結果一致性，不重新估計準確率；未建立全新 venv 或在另一台電腦驗證。

## detection-only-v1.0.0 Release

- 專屬資源 ZIP 僅含 30 個偵測 checkpoint，分為四段，總計 4,031,818,338 bytes；各模型 SHA256 與既有版本相同。沒有打包 Qwen、GGUF、mmproj 或 llama.cpp。
- 在全新的原始碼資料夾，使用本版四個分段實際完成整包 SHA256 核對、解壓、路徑重建，以及 15 種產品與全部權重的完整校驗。
- 從這次解壓安裝的模型執行 bottle、capsule、pill 各一張 OK／NG，共六張；判決與保存的原版結果相同，mask 逐像素一致，未匯入分類模組。
- 13 項回歸測試通過；本機封裝與安裝紀錄位於 `outputs/github_detection_release_20261018/`。
- README 明列完整版與精簡版的功能差異、分支、Release 與獨立資料夾安裝方式。
