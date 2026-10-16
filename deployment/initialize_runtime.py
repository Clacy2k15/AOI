"""Materialize machine-specific metadata from portable templates, never edit weights."""
import hashlib
import json
from pathlib import Path

TOKEN = '@AOI_ROOT@'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def materialize(value, root):
    if isinstance(value, str):
        return value.replace(TOKEN, str(root.resolve()))
    if isinstance(value, list):
        return [materialize(v, root) for v in value]
    if isinstance(value, dict):
        return {k: materialize(v, root) for k, v in value.items()}
    return value


def initialize(root):
    import yaml
    templates = json.loads((root / 'deployment/path_templates.json').read_text(encoding='utf-8'))
    for relative, value in templates['files'].items():
        destination = inside(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(materialize(value, root), ensure_ascii=False, indent=2), encoding='utf-8')
    for relative, key in templates['bundle_configs'].items():
        path = inside(root, relative)
        settings = yaml.safe_load(path.read_text(encoding='utf-8'))
        settings[key]['bundle_sha256'] = sha(inside(root, settings[key]['bundle']))
        path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding='utf-8')
    # These three redistributable CUDA runtime libraries already come with pinned PyTorch.
    import shutil
    import torch
    library = Path(torch.__file__).parent / 'lib'
    for name in ('cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll'):
        target = root / 'models/llama.cpp/b11609' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library / name, target)
    print('Runtime paths initialized for:', root)


if __name__ == '__main__':
    initialize(Path(__file__).resolve().parents[1])
