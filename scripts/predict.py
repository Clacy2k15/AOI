"""單張／資料夾偵測：只產生 OK／NG、mask 及疊圖。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi.common import ROOT, configure_runtime, load_config, output_path
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/aoi.yaml')
    parser.add_argument('--inference-config', help='Override the product inference config')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', help='Defaults to output_root/predictions')
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    configure_runtime(config)
    from aoi.pipeline import Pipeline, save_records
    source = Path(args.input).resolve()
    paths = ([source] if source.is_file() and source.suffix.lower() in EXTENSIONS else
             sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
             if source.is_dir() else [])
    if not paths:
        parser.error('No supported input images: ' + str(source))
    destination = args.output or output_path(config, 'predictions')
    pipeline = Pipeline(config, args.inference_config)
    records = []
    try:
        for path in paths:
            record = pipeline.predict(path, destination)
            records.append(record)
            print(f"{path.name} {record['decision'] or 'ERROR'} {record['error'] or ''}", flush=True)
    finally:
        pipeline.close()
    save_records(records, destination)
    return 1 if any(record['error'] for record in records) else 0


if __name__ == '__main__':
    sys.exit(main())
