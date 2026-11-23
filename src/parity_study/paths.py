"""Separate durable project notebooks from generated scratch outputs."""
import json
from pathlib import Path

OUTPUT_ROOT = Path('/home/users/h/hes/scratch/projects/cosmicPV')


def independent_ic_output(report_root):
    """Return the completed independent-IC report explicitly activated for display."""
    report_root = Path(report_root)
    pointer = report_root/'active_independent_ic.json'
    if not pointer.exists():
        return None
    relative = Path(json.loads(pointer.read_text())['directory'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Independent-IC report must stay within its report folder')
    return report_root/relative


def selection_output(report_root):
    """Locate the explicitly activated selection report, keeping old versions intact."""
    report_root = Path(report_root)
    pointer = report_root/'active_selection.json'
    if pointer.exists():
        directory = json.loads(pointer.read_text())['directory']
        relative = Path(directory)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Selection report directory must stay within its report folder')
        return report_root/relative
    return report_root/'selections'


def project_for_run(run):
    """Frozen runs record their project independently of their data location."""
    manifest = Path(run)/'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Cannot locate the project without {manifest}')
    return Path(json.loads(manifest.read_text())['project'])


def expose_output(project, target):
    """Expose scratch folders under results/saved; keep notebooks in results."""
    target = Path(target)
    parent = Path(project)/'results'
    if target.is_dir():
        parent = parent/'saved'
    alias = parent/target.name
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        if alias.resolve() != target.resolve():
            raise FileExistsError(f'Output link points elsewhere: {alias}')
    elif alias.exists():
        if alias.resolve() != target.resolve():
            raise FileExistsError(f'Refusing to replace an existing output: {alias}')
    else:
        alias.symlink_to(target, target_is_directory=target.is_dir())
    return alias
