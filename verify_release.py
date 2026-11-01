#!/usr/bin/env python3
"""Validate the sealed artifact; standard library only. No writes or network."""
from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parent

def main()->None:
    manifest=ROOT/'SHA256SUMS.txt'
    if not manifest.is_file(): raise SystemExit('STOP: SHA256SUMS.txt missing')
    failures=[];count=0;seen=set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        m=re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
        if not m:raise SystemExit('STOP: malformed checksum line')
        digest,name=m.groups();pp=PurePosixPath(name)
        if pp.is_absolute() or '..' in pp.parts or name in seen:raise SystemExit('STOP: unsafe or duplicate path')
        seen.add(name);p=ROOT.joinpath(*pp.parts)
        if p.is_symlink() or not p.is_file(): failures.append(name+': missing/unsafe');continue
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=digest:failures.append(name+': checksum mismatch')
        count+=1
    if failures:raise SystemExit('RELEASE_VERIFY_FAILED\n'+'\n'.join(failures))
    extras=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if '.git' in rel.parts or '__pycache__' in rel.parts or p.suffix=='.pyc':continue
        if p.is_file() and rel.as_posix() not in seen and rel.as_posix()!='SHA256SUMS.txt':extras.append(rel.as_posix())
    if extras:raise SystemExit('STOP: unmanifested payload files\n'+'\n'.join(extras))
    version=(ROOT/'VERSION').read_text().strip()
    meta=json.loads((ROOT/'.zenodo.json').read_text())
    cff=(ROOT/'CITATION.cff').read_text()
    cffv=re.search(r'^version:\s*["\x27]?([^"\x27\s]+)',cff,re.M)
    if not cffv or cffv[1]!=version or meta.get('version')!=version:raise SystemExit('STOP: version disagreement')
    if meta.get('upload_type')!='software' or not meta.get('creators') or not meta.get('license'):
        raise SystemExit('STOP: incomplete Zenodo metadata')
    if 'doi' in meta or re.search(r'^doi:',cff,re.M):
        print('NOTE: a DOI is present; confirm it is the actual assigned RASC identifier.')
    for forbidden in ['rasc_midpoint_R3.tex','rasc_midpoint_R4.tex','rasc_midpoint_R3.pdf','rasc_midpoint_R4.pdf',
                      'rasc_midpoint_JNM.tex','manuscript_JNM.pdf','cover_letter_JNM.pdf',
                      'cover_letter.tex','cover_letter.txt']:
        if any(p.name==forbidden for p in ROOT.rglob(forbidden)):raise SystemExit('STOP: manuscript in scientific release')
    print(json.dumps({'status':'PASS','payload_files':count,'version':version,
        'scope':'Payload integrity and basic metadata; no simulation or global-error certification.'},indent=2))
if __name__=='__main__':main()
