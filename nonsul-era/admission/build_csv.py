"""schools/*.yaml -> summary.csv(학교×연도 1줄), changes.csv(연도 간 자동 비교). 실행: python3 build_csv.py"""
import csv, glob, os, yaml
H = os.path.dirname(os.path.abspath(__file__))
def j(x):
    if x is None: return ''
    if isinstance(x, dict): return '; '.join(f'{k} {v}' for k, v in x.items())
    if isinstance(x, list): return ' | '.join(j(i) if not isinstance(i, dict) else ', '.join(f'{k}={v}' for k, v in i.items() if v not in (None, '')) for i in x)
    return str(x)
rows, ch = [], []
for f in sorted(glob.glob(os.path.join(H, 'schools', '*.yaml'))):
    d = yaml.safe_load(open(f, encoding='utf-8'))
    prev = None
    for y in d['years']:
        ex = y.get('exam') or []
        r = {'code': d['code'], 'school': d['school'], 'year': y['year'], 'doc': y.get('doc') or '', 'missing': y.get('missing') or '',
             'nonsul': y.get('nonsul', ''), 'jeonhyeong': y.get('jeonhyeong') or '', 'total_quota': y.get('total_quota') or '',
             'by_track': j(y.get('by_track')), 'units_n': len(y.get('units') or []), 'ratio': j(y.get('ratio')),
             'csat_min': j([{'a': c.get('applies'), 'r': c.get('rule')} for c in (y.get('csat_min') or [])]),
             'schedule': j(y.get('schedule')),
             'exam': j([{k: e.get(k) for k in ('type', 'minutes', 'items', 'answer_form', 'subjects', 'science')} for e in ex]),
             'url': y.get('url') or '', 'pages': y.get('pages') or ''}
        rows.append(r)
        if prev and not r['missing'] and not prev['missing']:
            for k in ('nonsul', 'total_quota', 'ratio', 'csat_min', 'exam'):
                if str(prev[k]) != str(r[k]) and r[k] not in ('', None):
                    ch.append({'code': d['code'], 'school': d['school'], 'from': prev['year'], 'to': r['year'], 'field': k, 'before': prev[k], 'after': r[k]})
        if not r['missing']: prev = r
for name, data in (('summary.csv', rows), ('changes.csv', ch)):
    with open(os.path.join(H, name), 'w', newline='', encoding='utf-8-sig') as fo:
        if data:
            w = csv.DictWriter(fo, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print(len(rows), 'rows,', len(ch), 'changes')
