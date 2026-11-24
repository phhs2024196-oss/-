import pypdf, re, json
r = pypdf.PdfReader('body.pdf'); m = {}
for i, p in enumerate(r.pages):
    t = p.extract_text()
    for x in re.finditer(r'@\s*@([\d\s]+?)@\s*@', t):
        n = re.sub(r'\s', '', x.group(1)); m.setdefault(n, i+1)
        # marker near top of page => previous page holds end of old page n-1
        m.setdefault('top'+n, x.start() < 15)
out = {}
keys = sorted(int(k) for k in m if k.isdigit() and int(k) < 100)
for a, b in zip(keys, keys[1:]):
    n = a
    out[str(n)] = m[str(n)]
    nxt = m[str(b)]
    out['E'+str(n)] = max(m[str(n)], nxt - 1 if m['top'+str(b)] else nxt)
out['121'] = m['121']
for k in m:
    if k.isdigit() and 200 < int(k) < 300: out[k] = m[k]
json.dump(out, open('pmap.json','w')); print(len(r.pages))
