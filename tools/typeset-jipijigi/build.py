import re, sys, markdown, html
B = '/mnt/project-files/books/jipijigi/'
import os as _o
FILES = _o.environ['SRC'].split(',') if _o.environ.get('SRC') else [B + '총론_1-20쪽_초고_v1.md', B + '총론_21-70쪽_초고_v1.md']
MARK = re.compile(r'\s*\[(확인|데이터 없음)[^\]]*\]')

def clean_line(l):
    if l.lstrip().startswith('|'):
        cells = l.split('|')
        out = []
        for c in cells:
            if MARK.fullmatch(' ' + c.strip()) and c.strip():
                out.append(' – ')
            else:
                out.append(MARK.sub('', c))
        return '|'.join(out)
    return MARK.sub('', l)

def load(p):
    t = open(p, encoding='utf-8').read()
    t = t.split('\n## 검수용 수치 목록')[0]
    t = t[re.search(r'^〔p\.\d+〕', t, re.M).start():]          # drop title + editor memos
    return [clean_line(l) for l in t.split('\n')]

lines = load(FILES[0]) + [''] + load(FILES[1])

# split into pages by 〔p.N〕
pages, cur, num = [], [], None
for l in lines:
    m = re.match(r'^〔p\.(\d+)〕\s*$', l.strip())
    if m:
        if num is not None: pages.append((num, cur))
        num, cur = int(m.group(1)), []
    else:
        cur.append(l)
pages.append((num, cur))

def circ(c):
    o = ord(c)
    return o - 0x2460 + 1 if 0x2460 <= o <= 0x2473 else o - 0x3251 + 21

def pageref(t):
    # 8부 type index table: "(NN)" -> page of the first circled type in that group
    def tl(line):
        if not line.startswith('|') or not re.search('[①-⑳㉑-㉘]', line): return line
        return re.sub(r'([①-⑳㉑-㉘])((?:(?!\(\d+\))[^|])*?)\((\d+)\)', lambda m: f'{m.group(1)}{m.group(2)}({{{{P:{200+circ(m.group(1))}}}}})', line)
    t = '\n'.join(tl(x) for x in t.split('\n'))
    t = re.sub(r'\((\d+)~(\d+)쪽\)', lambda m: f'({{{{P:{m.group(1)}}}}}~{{{{PE:{m.group(2)}}}}}쪽)', t)
    t = re.sub(r'\((\d+)쪽\)', lambda m: f'({{{{P:{m.group(1)}}}}}쪽)' if int(m.group(1))<=70 else m.group(0), t)
    t = re.sub(r'(\d+)쪽( 표| 참| 예| 이후|부터)', lambda m: f'{{{{P:{m.group(1)}}}}}쪽{m.group(2)}', t)
    t = re.sub(r'(\d+)쪽(까지| 안)', lambda m: f'{{{{PE:{m.group(1)}}}}}쪽{m.group(2)}', t)
    t = t.replace('6부 ({{P:21}}쪽)', '6부 ({{P:121}}쪽)')
    return t

md = markdown.Markdown(extensions=['tables', 'sane_lists'])

def render(block):
    txt = '\n'.join(block)
    txt = pageref(txt)
    txt = re.sub(r'^\s*---\s*$', '', txt, flags=re.M)        # hr = editorial separators
    txt = re.sub(r'^- \[ \] ', '- ☐ ', txt, flags=re.M)
    # python-markdown needs a blank line before a list
    L = txt.split('\n'); o = []
    isl = lambda x: re.match(r'^(>\s*)?(\s*[-*]|\s*\d+\.)\s', x) is not None
    for i, x in enumerate(L):
        if i and x.startswith('|') and L[i-1].strip() and not L[i-1].startswith('|'):
            o.append('')
        if i and isl(x) and L[i-1].strip() not in ('', '>') and not isl(L[i-1]):
            o.append('>' if x.startswith('>') else '')
        o.append(x)
    txt = '\n'.join(o)
    md.reset()
    h = md.convert(txt)
    # 이것만 기억하세요 boxes
    h = re.sub(r'<blockquote>\s*<p><strong>이것만 기억하세요</strong></p>',
               '<blockquote class="remember"><p class="rem-title">이것만 기억하세요</p>', h)
    h = re.sub(r'<blockquote>\s*<p><strong>(결론 먼저\.|결론\.)</strong>',
               r'<blockquote class="lead"><p><strong>\1</strong>', h)
    h = re.sub(r'<blockquote>\s*<p><strong>(잠깐,)', r'<blockquote class="notice"><p><strong>\1', h)
    h = h.replace('[QR]', '<span class="qr"><span>QR</span><small>문일지십<br>무료 진단고사</small></span>')
    h = re.sub(r'<p><strong>(\[표 [^\]]+\][^<]*)</strong></p>', r'<p class="cap">\1</p>', h)
    for k,c in (('핵심','key'),('주의','warn'),('풀이 팁','tip'),('생각해 볼 질문','ask')):
        h = re.sub(r'<blockquote>\s*<p><strong>('+k+r'[.:]?)</strong>', r'<blockquote class="bx bx-'+c+r'"><p><strong>\1</strong>', h)
    h = re.sub(r'<h2>(\d+장\.|\d+부\.|부록 [A-Z][.:]?)\s*', r'<h2><span class="no">\1</span>', h)
    return h

import visuals, os
body = []
for n, blk in pages:
    h = visuals.apply(n, render(blk))
    if os.environ.get('V2'): h = visuals.apply_v2(n, h)
    h = h.lstrip()
    h = re.sub(r'^(<[a-z0-9]+[^>]*>)', lambda m: m.group(1) + f'<span class="pm">@@{n}@@</span>', h, count=1)
    body.append(f'<div class="op" data-p="{n}">{h}</div>')
body.append(f'<div class="op"><span class="pm">@@{pages[-1][0]+1}@@</span></div>')
body = [re.sub(r'<h3>((?:<span class="pm">@@\d+@@</span>)?)([①-⑳㉑-㉘])', lambda m: f'<h3>{m.group(1)}<span class="pm">@@{200+circ(m.group(2))}@@</span>{m.group(2)}', b) if int((re.search(r'data-p="(\d+)"', b) or [0,'0'])[1]) >= 37 else b for b in body]
body = [b.replace('<span class="no">6부.</span>', '<span class="no"><span class="pm">@@121@@</span>6부.</span>') for b in body]

css = open(sys.argv[1], encoding='utf-8').read()
import json, os
pmap = json.load(open('pmap.json')) if os.path.exists('pmap.json') else {}
def sub(m):
    k, n = m.group(1), m.group(2)
    if not pmap: return '00'
    return str(pmap[n]) if k == 'P' else str(pmap['E' + n])
body = [re.sub(r'\{\{(PE|P):(\d+)\}\}', sub, b) for b in body]
if os.environ.get('FINAL'):
    body = [re.sub(r'<span class="pm">@@\d+@@</span>', '', b) for b in body]
out = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{css}</style></head><body>{"".join(body)}</body></html>'
open(sys.argv[2], 'w', encoding='utf-8').write(out)
print('pages in manuscript:', len(pages), [p[0] for p in pages][:3], '...', pages[-1][0])
