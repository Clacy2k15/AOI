# 安裝階段的模型分段下載、SHA256 核對、合併及安全解壓工具；只使用 Python 標準函式庫。
# release-manifest.json 固定模型來源版本、每段大小／雜湊、整包雜湊及解壓檔案清單。
# 這個下載工具不啟用推論用的離線攔截器；安裝完成後才由推論程序限制對外網路。

"""Download pinned GitHub Release parts (or use local parts), verify and extract."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from initialize_runtime import inside, sha


# 只解壓 manifest 明確列出的檔案，拒絕重複項目、越界路徑、符號連結及大小不一致的成員。
# 每個檔案先寫 .installing 並核對 SHA256，通過後才替換正式目的檔，失敗時清掉暫存檔。
def extract_verified(archive_path, root, expected):
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        # 解壓清單必須與 manifest 完全相等，不能只檢查「至少包含」必要模型。
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Archive contents differ from the release manifest')
        for info in archive.infolist():
            target = inside(root, info.filename)
            record = expected[info.filename]
            if info.file_size != record['size'] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Invalid archive member: ' + info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            # 先寫臨時檔，避免下載／解壓中斷或校驗失敗時破壞使用者已安裝的同名權重。
            temporary = target.with_name(target.name + '.installing')
            try:
                with archive.open(info) as source, temporary.open('wb') as destination:
                    shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                if sha(temporary) != record['sha256']:
                    raise ValueError('File checksum mismatch: ' + info.filename)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


# 可從本機分段資料夾安裝，或從固定 GitHub Release 下載；每段都先驗大小與雜湊。
# 依清單順序串流合併 ZIP，核對整包後安全解壓；保留下載分段供重跑，刪除臨時整包 ZIP。
def install(root, source_directory=None):
    manifest = json.loads((root / 'release-manifest.json').read_text(encoding='utf-8'))
    cache = root / '.downloads'
    cache.mkdir(exist_ok=True)
    parts = []
    for record in manifest['parts']:
        name = record['name']
        # 分段名稱只能是單純檔名；不可透過名稱帶入目錄、反斜線或絕對路徑。
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid part name')
        path = source_directory / name if source_directory else cache / name
        if not path.exists() or path.stat().st_size != record['size'] or sha(path) != record['sha256']:
            # 使用者選本機安裝時，缺段或損壞就明確失敗；不暗中轉成線上下載。
            if source_directory:
                raise ValueError('Local release part missing or corrupt: ' + str(path))
            # 來源由固定 repository／tag／name 組成，而非抓 latest，避免日後新版模型混入舊設定。
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
                # 每次只讀 8 MiB；模型包約數 GB，不能一次讀進 RAM。合併順序以 manifest 為準。
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
                    destination.write(block)
    # 單段正確仍需核對整包，確保分段順序與組合也完全符合發佈內容。
    if digest.hexdigest() != manifest['archive_sha256']:
        archive.unlink(missing_ok=True)
        raise ValueError('Combined archive checksum mismatch')
    extract_verified(archive, root, manifest['files'])
    archive.unlink()
    print('All runtime files verified and extracted.', flush=True)


# --from-dir 指向所有 Release 分段所在資料夾；省略時才使用 GitHub 網路下載。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, help='Local directory containing all Release parts')
    args = parser.parse_args()
    install(Path(__file__).resolve().parents[1], args.from_dir)
