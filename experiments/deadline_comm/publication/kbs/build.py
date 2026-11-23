from pathlib import Path
import subprocess,json,re,hashlib,zipfile
P=Path(__file__).resolve().parent
def main():
 with (P/'build.log').open('w') as f:
  for c in [['pdflatex','-interaction=nonstopmode','-halt-on-error','kbs_main.tex'],['bibtex','kbs_main'],['pdflatex','-interaction=nonstopmode','-halt-on-error','kbs_main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','kbs_main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','kbs_main.tex']]:subprocess.run(c,cwd=P,stdout=f,stderr=subprocess.STDOUT,check=True)
 log=(P/'kbs_main.log').read_text();txt=subprocess.check_output(['pdftotext','-layout',str(P/'kbs_main.pdf'),'-'],text=True)
 for bad in ['undefined references','multiply defined','Missing character','Overfull']:assert bad not in log,bad
 assert 'Under review as submission to TMLR' not in txt
 assert 'Yeong-woo Kung' in txt and 'Young-guk Ha' in txt
 manifest=json.loads((P/'source_manifest.json').read_text());assert hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest()==manifest['source_sha256']
 assert hashlib.sha256((P.parent/'tmlr/tmlr_main.pdf').read_bytes()).hexdigest()==manifest['tmlr_pdf_sha256']
 state=dict(pages=len([x for x in txt.split('\f') if x.strip()]),numerical_citations=bool(re.search(r'\[1[,\]]',txt)),tmlr_preserved=True,compiled=True,author_details_pending=True,journal_guide_fully_verified=False)
 (P/'validation.json').write_text(json.dumps(state,indent=2));print(state)
 with zipfile.ZipFile(P/'kbs_source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in P.rglob('*'):
   if f.suffix in ['.tex','.bib','.bbl','.cls','.bst','.sty'] or f.parent.name=='figures' or f.name in ['Highlights.txt','AUTHOR_INFORMATION_PENDING.txt','README.md','build.py']:z.write(f,f.relative_to(P))
if __name__=='__main__':main()
