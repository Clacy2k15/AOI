# 命令列推論入口：接受產品設定與單張影像／資料夾，交由共用 Pipeline 執行。
# 範例：python scripts/predict.py --config configs/products/bottle/aoi.yaml --input D:/images。
# 新圖片預設執行真實模型；--detectors-only 略過 Qwen，--reuse-dev-results 則只允許歷史固定 dev。

"""Single image / folder CLI over the shared pipeline."""
import argparse
import sys
from pathlib import Path

# 支援從 scripts/ 直接執行，先把專案根目錄加到 import 搜尋路徑，無需把 aoi 安裝成套件。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi.common import ROOT, configure_runtime, load_config, output_path

if __name__ == "__main__":
    # 解析 CLI 參數；產品由 --config 決定，不從圖片檔名或瑕疵資料夾名稱猜測。
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aoi.yaml")
    parser.add_argument("--inference-config", help="Defaults to the product AOI config's inference_config")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Defaults to the product output_root/predictions")
    parser.add_argument("--detectors-only", action="store_true", help="Skip Qwen while retaining NG")
    parser.add_argument('--reuse-dev-results',action='store_true',help='Cache-only: identical fixed dev images/settings; no model inference')
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    args.output = args.output or str(output_path(config, "predictions"))
    # 在 import Pipeline／Torch 之前先設定快取與執行緒相關環境參數。
    configure_runtime(config)
    from aoi.pipeline import Pipeline, save_records

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        parser.error("Input path does not exist: " + str(input_path))
    # 資料夾會遞迴收集支援的影像副檔名並排序，確保批次順序可追溯。
    paths = [input_path] if input_path.is_file() else sorted(p for p in input_path.rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        parser.error("No supported images in the input folder")
    # 整批只建立一次模型；split 標記區分一般新圖推論與受限制的 dev 快取模式。
    pipeline = Pipeline(config, args.inference_config, classify=not args.detectors_only,
                        reuse_dev_results=args.reuse_dev_results,
                        split='dev' if args.reuse_dev_results else 'inference')
    records = []
    try:
        for path in paths:
            result = pipeline.predict(path, args.output)
            records.append(result)
            print(f"{result['sample_id']} {result['decision']} {result['defect_type']} {result['classification_status']}", flush=True)
    # 不論中途哪張影像發生錯誤，都嘗試釋放模型，避免命令結束前遺留顯存引用。
    finally:
        pipeline.close()
    save_records(records, args.output)
    # REVIEW 或分類失敗回傳非零退出碼，讓外部批次工作能辨識此次執行未完全成功。
    sys.exit(1 if any(record["decision"] == "REVIEW" or record["classification_status"] == "failed" for record in records) else 0)
