#!/usr/bin/env python3
"""Build the client ZIP, excluding environments, scratch data and generated model caches."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE_ROOT=Path('Team19_Automated_Handover')


def included(path):
    parts=path.relative_to(ROOT).parts
    if any(p in ('__pycache__','.pytest_cache','.venv','.git','.DS_Store','build','dist',
                  '__MACOSX','.ipynb_checkpoints')
           or p.endswith('.egg-info') or p.startswith('._') for p in parts):
        return False
    if path.suffix in ('.pyc','.pyo','.log'):
        return False
    if parts[0]=='outputs':
        return parts==('outputs','imputation-study','runs','README.md')
    if parts[0]=='reports':
        return len(parts)>1 and (parts[1]=='published' or parts[1]=='selection_audit.json')
    return path.name!='PACKAGE_MANIFEST.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    expected=json.loads((ROOT/'ORIGINAL_SOURCES.json').read_text())
    for key in ('original_prediction_modules','original_input_files','original_result_csvs'):
        for name,digest in expected[key].items():
            actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual!=digest:
                raise SystemExit(f'Original scientific source or data changed: {name}')
    if not (ROOT/'reports/published/report.html').is_file():
        raise SystemExit('Generate the report before packaging: python run_study.py')
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and included(p)]
    manifest={'package':'Team19 automated imputation handover','version':'5.1.0',
              'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.out,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for file in files+[ROOT/'PACKAGE_MANIFEST.json']:
            z.write(file,ARCHIVE_ROOT/file.relative_to(ROOT))
    print(f'Packaged {len(files)+1} files: {args.out} ({args.out.stat().st_size/1e6:.1f} MB)')


if __name__=='__main__':
    main()
