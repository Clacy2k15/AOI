"""Download pinned GitHub Release parts (or use local parts), verify and extract."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from initialize_runtime import inside, sha


def extract_verified(archive_path, root, expected):
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Archive contents differ from the release manifest')
        for info in archive.infolist():
            target = inside(root, info.filename)
            record = expected[info.filename]
            if info.file_size != record['size'] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Invalid archive member: ' + info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.installing')
            try:
                with archive.open(info) as source, temporary.open('wb') as destination:
                    shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                if sha(temporary) != record['sha256']:
                    raise ValueError('File checksum mismatch: ' + info.filename)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def install(root, source_directory=None):
    manifest = json.loads((root / 'release-manifest.json').read_text(encoding='utf-8'))
    cache = root / '.downloads'
    cache.mkdir(exist_ok=True)
    parts = []
    for record in manifest['parts']:
        name = record['name']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid part name')
        path = source_directory / name if source_directory else cache / name
        if not path.exists() or path.stat().st_size != record['size'] or sha(path) != record['sha256']:
            if source_directory:
                raise ValueError('Local release part missing or corrupt: ' + str(path))
            url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}/{name}"
            print('Downloading', name, flush=True)
            temporary = path.with_suffix('.downloading')
            request = urllib.request.Request(url, headers={'User-Agent': 'AOI-installer'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            if temporary.stat().st_size != record['size'] or sha(temporary) != record['sha256']:
                temporary.unlink(missing_ok=True)
                raise ValueError('Release part checksum mismatch: ' + name)
            temporary.replace(path)
        parts.append(path)
    archive = cache / 'runtime.zip'
    digest = hashlib.sha256()
    print('Assembling and verifying runtime archive...', flush=True)
    with archive.open('wb') as destination:
        for part in parts:
            with part.open('rb') as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
                    destination.write(block)
    if digest.hexdigest() != manifest['archive_sha256']:
        archive.unlink(missing_ok=True)
        raise ValueError('Combined archive checksum mismatch')
    extract_verified(archive, root, manifest['files'])
    archive.unlink()
    print('All runtime files verified and extracted.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, help='Local directory containing all Release parts')
    args = parser.parse_args()
    install(Path(__file__).resolve().parents[1], args.from_dir)
