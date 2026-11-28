"""Build the JAAMAS submission documents using the unchanged official class."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess, json, re
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--overleaf', action='store_true', help='Also package the main manuscript as a single-document Overleaf project.')
args = parser.parse_args()
P=Path(__file__).resolve().parent
names=['jaamas_main','jaamas_supplement','information_sheet','cover_letter','statements_and_declarations','title_page']
def run(args):
 r=subprocess.run(args,cwd=P,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 if r.returncode: raise RuntimeError(r.stdout[-5000:])
 return r.stdout
results={}
for name in names:
 run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'])
 if name.startswith('jaamas_'): run(['bibtex',name])
 for _ in range(3):run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'])
 log=(P/(name+'.log')).read_text()
 problems=[x for x in log.splitlines() if any(y in x for y in ['Overfull','undefined','multiply defined','Float too large','Missing character'])]
 info=run(['pdfinfo',name+'.pdf'])
 results[name]={'pages':int(re.search(r'Pages:\s+(\d+)',info)[1]),'warnings':problems}
print(json.dumps(results,indent=2))
(P/'build_validation.json').write_text(json.dumps(results,indent=2)+'\n')
if any(v['warnings'] for v in results.values()):raise SystemExit('Resolve layout/reference warnings before packaging')
if not 1<=results['information_sheet']['pages']<=2:raise SystemExit('Information sheet must be 1-2 pages')

if args.overleaf:
    manuscript = P / 'jaamas_main.tex'
    graphics = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', manuscript.read_text())
    names = ['sn-jnl.cls', 'journal-reading-layout.sty', 'sn-mathphys-num.bst', 'references.bib', *graphics]
    target = P / 'JAAMAS_Main_Overleaf.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        archive.write(manuscript, 'main.tex')
        for name in dict.fromkeys(names):
            archive.write(P / name, name)
    print(target)
