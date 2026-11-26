"""Build JAAMAS files and optionally a portable reviewer archive; never submits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['jaamas_main.tex','jaamas_supplement.tex','information_sheet.tex','cover_letter.tex','statements_and_declarations.tex','references.bib','sn-jnl.cls','sn-mathphys-num.bst','diagnostics.pdf','receiver_adaptation.pdf','seed_distribution.pdf','robustness.pdf','build.py','plot_diagnostics.py','figure_inputs.json']
PDFS={'jaamas_main.pdf':'Main_Manuscript.pdf','jaamas_supplement.pdf':'Supplementary.pdf','information_sheet.pdf':'Information_Sheet.pdf','cover_letter.pdf':'Cover_Letter.pdf','statements_and_declarations.pdf':'Statements_and_Declarations.pdf'}


def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 a=argparse.ArgumentParser();a.add_argument('--reviewer',action='store_true');a.add_argument('--skip-build',action='store_true');a.add_argument('--reviewer-output',default='dist/Reviewer_Archive.zip');args=a.parse_args()
 paper=ROOT/'paper';out=ROOT/'submission/jaamas';out.mkdir(parents=True,exist_ok=True)
 if not args.skip_build:subprocess.run([sys.executable,'build.py'],cwd=paper,check=True)
 for source,name in PDFS.items():shutil.copy2(paper/source,out/name)
 with zipfile.ZipFile(out/'Manuscript_Source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in SOURCES:z.write(paper/name,name)
  z.writestr('README.txt','JAAMAS editable sources. Primary manuscript: jaamas_main.tex. Supplement: jaamas_supplement.tex. Build all five PDFs: python build.py (pdflatex, BibTeX, pdfinfo required). The publisher class/style are unchanged.\n')
 files=[out/name for name in PDFS.values()]+[out/'Manuscript_Source.zip']+[p for p in out.glob('*.md')]
 (out/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(files)))
 with zipfile.ZipFile(out/'JAAMAS_Submission.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[out/'SHA256SUMS.txt']:z.write(p,p.name)
 if args.reviewer:
  target=Path(args.reviewer_output)
  if not target.is_absolute():target=ROOT/target
  if target.exists():raise FileExistsError(f'Preserve the existing archive or choose another output: {target}')
  target.parent.mkdir(parents=True,exist_ok=True)
  # Works from a downloaded ZIP too; reconstructed original-path inputs are excluded.
  allowed=['data','results','checkpoints','docs','experiments','models','tests','paper','scripts','submission']
  files=[]
  for directory in allowed:
   for p in (ROOT/directory).rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(ROOT);parts=rel.parts
    if any(v in parts for v in ['__pycache__','.pytest_cache','.venv','runs']):
     if parts[0] not in ('data','results','checkpoints'):continue
    if parts[0]=='experiments' and ('repro_assets' in parts or ('tmlr_revision' in parts and 'results' in parts) or ('external_biases' in parts and 'data' in parts)):continue
    if p.suffix in ['.pyc','.aux','.blg','.log','.out','.abs','.spl','.toc']:
     # Preserve published clean-install evidence; exclude LaTeX/runtime scratch.
     if not (parts[0]=='docs' and p.suffix=='.log'):continue
    if p==target or p.name=='Reviewer_Archive.zip':continue
    files.append(p)
  files += [p for p in ROOT.iterdir() if p.is_file() and p.name in ['README.md','requirements.txt','constraints-cpu-verified.txt','.gitignore']]
  with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   lines=[]
   for p in sorted(set(files)):
    rel=p.relative_to(ROOT).as_posix();z.write(p,rel);lines.append(f'{digest(p)}  {rel}\n')
   z.writestr('REVIEWER_SHA256SUMS.txt',''.join(lines))
  print(json.dumps({'reviewer_archive':str(target),'files':len(lines),'bytes':target.stat().st_size,'sha256':digest(target)}))
 print(out)
if __name__=='__main__':main()
