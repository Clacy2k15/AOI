"""Check the installed inference assets and all 15 configurations without training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from initialize_runtime import inside, sha


def check(root, full=False):
    import torch
    from aoi.common import load_config, load_inference
    from aoi.inference import training_summary
    from aoi.capsule_runtime import validate_deployment as capsule
    from aoi.pill_runtime import validate_deployment as pill
    manifest = json.loads((root / 'release-manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        path = inside(root, name)
        if not path.is_file() or path.stat().st_size != record['size']:
            raise ValueError('Missing/incomplete runtime asset: ' + name)
        if full and sha(path) != record['sha256']:
            raise ValueError('Runtime asset checksum mismatch: ' + name)
    for product, row in json.loads((root / 'outputs/expansion/product_registry.json').read_text(encoding='utf-8'))['products'].items():
        config = load_config(row['config'])
        settings = load_inference(config, require_calibrated=True)
        inside(root, settings['normal_reference']).read_bytes()
        training_summary(config)
        if product == 'capsule':
            capsule(config, settings)
        if product == 'pill':
            pill(config, settings)
        print('Configuration OK:', product)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Install a compatible NVIDIA driver; this release requires CUDA inference.')
    for name in ('cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll'):
        if not (root / 'models/llama.cpp/b11609' / name).is_file():
            raise FileNotFoundError(name + '; rerun initialize_runtime.py')
    print('GPU:', torch.cuda.get_device_name(0))
    print('Runtime check passed. Start with start-demo.ps1.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Hash every asset again')
    check(Path(__file__).resolve().parents[1], parser.parse_args().full)
