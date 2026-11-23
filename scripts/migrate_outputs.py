#!/usr/bin/env python3
"""Copy outputs to scratch, verify SHA-256, then replace originals with links."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import stat
import subprocess
from zoneinfo import ZoneInfo
from parity_study.paths import OUTPUT_ROOT
from parity_study.storage import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('20261121_baseline','20261121_notebook_current','basic','intro_disp_PV','notebook_jobs.json')


def inventory(path, *, hashes):
    entries = {}
    for p in [path, *sorted(path.rglob('*'))] if path.is_dir() else [path]:
        key = str(p.relative_to(path)) if p != path else '.'
        info = p.lstat()
        if p.is_symlink():
            entries[key] = dict(type='link',target=str(p.readlink()))
        elif p.is_file():
            entries[key] = dict(type='file',size=info.st_size,
                                **({'sha256':digest(p)} if hashes else {'mtime_ns':info.st_mtime_ns}))
        elif p.is_dir():
            entries[key] = dict(type='directory')
        else:
            raise ValueError(f'Unsupported output type: {p}')
        if p.lstat().st_mtime_ns != info.st_mtime_ns:
            raise RuntimeError(f'Output changed during verification: {p}')
    return entries


def main():
    now = datetime.now(ZoneInfo('Europe/Zurich'))
    stamp = now.strftime('%Y%m%d_%H%M%S')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = OUTPUT_ROOT/f'.migration-{stamp}'
    retired_root = ROOT/'results'/f'.migration-{stamp}'
    stage.mkdir()
    retired_root.mkdir()
    receipt = dict(created_at=now.isoformat(timespec='seconds'),destination=str(OUTPUT_ROOT),outputs={})
    for name in NAMES:
        source, target = ROOT/'results'/name, OUTPUT_ROOT/name
        if source.is_symlink() and source.resolve()==target.resolve():
            print(f'Already migrated: {name}', flush=True)
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Destination exists; leaving both locations intact: {target}')
        before = inventory(source, hashes=False)
        expected = inventory(source, hashes=True)
        staged = stage/name
        print(f'Copying and verifying: {name}', flush=True)
        subprocess.run(['rsync','-a','--',str(source),str(stage)+'/'],check=True)
        if inventory(staged, hashes=True) != expected or inventory(source, hashes=False) != before:
            raise RuntimeError(f'Verification failed or source changed: {name}; original retained')
        staged.rename(target)
        retired = retired_root/name
        source.rename(retired)
        try:
            source.symlink_to(target,target_is_directory=target.is_dir())
            if inventory(retired, hashes=False) != before:
                raise RuntimeError(f'Source changed before switch: {name}')
        except BaseException:
            if source.is_symlink():
                source.unlink()
            retired.rename(source)
            raise
        # Delete only the verified duplicate, after the scratch copy and link exist.
        if retired.is_dir():
            shutil.rmtree(retired)
        else:
            retired.unlink()
        files = [v for v in expected.values() if v['type']=='file']
        receipt['outputs'][name] = dict(files=len(files),bytes=sum(v['size'] for v in files),
                                       sha256_verified=True)
        atomic_json(OUTPUT_ROOT/'storage_migration.json',receipt)
        print(f'Migrated: {name}; {len(files)} files verified', flush=True)
    stage.rmdir()
    retired_root.rmdir()
    print('Migration complete; notebooks remain in the project results directory.',flush=True)


if __name__ == '__main__':
    main()
