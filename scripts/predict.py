"""Single image / folder CLI over the shared pipeline."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi.common import ROOT, configure_runtime, load_config, output_path

if __name__ == "__main__":
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
    configure_runtime(config)
    from aoi.pipeline import Pipeline, save_records

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        parser.error("Input path does not exist: " + str(input_path))
    paths = [input_path] if input_path.is_file() else sorted(p for p in input_path.rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        parser.error("No supported images in the input folder")
    pipeline = Pipeline(config, args.inference_config, classify=not args.detectors_only,
                        reuse_dev_results=args.reuse_dev_results,
                        split='dev' if args.reuse_dev_results else 'inference')
    records = []
    try:
        for path in paths:
            result = pipeline.predict(path, args.output)
            records.append(result)
            print(f"{result['sample_id']} {result['decision']} {result['defect_type']} {result['classification_status']}", flush=True)
    finally:
        pipeline.close()
    save_records(records, args.output)
    sys.exit(1 if any(record["decision"] == "REVIEW" or record["classification_status"] == "failed" for record in records) else 0)
