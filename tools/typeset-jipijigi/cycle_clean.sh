set -e
cp pmap.json pmap_old.json 2>/dev/null || true
rm -f pmap.json
for i in 1 2 3; do python3 build.py clean.css body.html >/dev/null; BODY=body.html node render_clean.js; cp body_clean.pdf body.pdf; python3 pmap.py; done
FINAL=1 python3 build.py clean.css body_clean.html >/dev/null; node render_clean.js
python3 -c "
import pypdf
w=pypdf.PdfWriter()
for f in ['cover_clean.pdf','body_clean.pdf']:
  for p in pypdf.PdfReader(f).pages: w.add_page(p)
w.add_metadata({'/Title':'지피지기(논술 총론)','/Author':'논술의 시대'})
w.write('clean_full.pdf'); print('pages',len(w.pages))"
