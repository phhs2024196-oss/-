#!/usr/bin/env python3
"""raw/<BOOK>/pXXX-YYY.yaml -> items/<BOOK>.yaml (해설 병합) + all.jsonl + stats.md"""
import yaml, json, re, glob, os, collections, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS = {
 'YSJ-GEO':('여상진 수리논술 2026 정규반 기하','정규반 교재'),
 'YSJ-DIF':('여상진 수리논술 2026 정규반 미분','정규반 교재'),
 'YSJ-INT':('여상진 수리논술 2026 정규반 적분','정규반 교재'),
 'YSJ-MATH':('여상진 수리논술 2026 정규반 수학 상·하','정규반 교재'),
 'YSJ-R23':('여상진 수리논술 정규반 23회','정규반 회차 교재'),
 'YSJ-MK19':('여상진 모의논술 19회','실전 모의'),'YSJ-MK21':('여상진 모의논술 21회','실전 모의'),
 'YSJ-MK22':('여상진 모의논술 22회','실전 모의'),'YSJ-MK23':('여상진 모의논술 23회','실전 모의'),
 'KKW-W7':('김기원 수리논술 겨울특강 7회차','특강 시험지'),
 'ANA-GEO':('ANALOG 수리논술 기하','개념+기출 교재'),'ANA-GEOS':('ANALOG 기하 손글씨해설편','해설집'),
 'ANA-PS':('ANALOG 수리논술 확률과 통계','개념+기출 교재'),'ANA-PSS':('ANALOG 확통 손글씨해설편','해설집'),
 'ANA-C2':('ANALOG 수리논술 수학Ⅱ·미적분 WORKBOOK','워크북'),
 'SJH-GEO':('서지현 수리논술 기하편','개념+기출 교재'),
 'WJS-YS':('우주설 수리논술 엑셀시어 연세대','학교별 대비 교재'),
 'SEMI':('셈이의 문제해결기법','문제해결 기법서'),
}
items=[]; problems=[]
for f in sorted(glob.glob(f'{R}/raw/*/p*.yaml')):
    book=f.split('/')[-2]; m=re.search(r'p(\d+)-(\d+)\.yaml',f); a,b=int(m[1]),int(m[2])
    try: d=yaml.safe_load(open(f)) or []
    except Exception as e: print('PARSE',f,e); continue
    for it in d:
        it.setdefault('book',book); it['_file']=os.path.relpath(f,R)
        if not (a<=int(it.get('page') or a)<=b): print('PAGE-OUT',it.get('id'),f)
        items.append(it)
def chap(s):
    m=re.match(r'\s*(\d{1,2})\b',str(s or '')); return m[1].zfill(2) if m else None
def key(it):
    lab=str(it.get('label') or ''); ch=chap(lab) or chap(it.get('chapter'))
    m=re.search(r'예제\s*(\d+)-(\d+)',lab)
    if m: return f'ex{m[1].zfill(2)}-{m[2]}'
    m=re.search(r'논제\s*(\d+)',lab)
    if m and ch: return f'{ch}-q{m[1]}'
    if it.get('book') in ('WJS-YS','SJH-GEO') or it.get('solves_book') in ('WJS-YS','SJH-GEO'):
        m=re.match(r'\s*(\d{1,2})(번)?(\s|$)',lab)
        if m: return f'n{int(m[1])}'
    return None
def semikey(it):
    lab=str(it.get('label') or ''); sec=str(it.get('section') or it.get('chapter') or '')
    m=re.search(r'연습문제\s*(\d+(?:\.\d+)+)-(\d+)',lab)
    if m: return f'x{m[1]}-{m[2]}'
    m=re.search(r'연습문제\s*(\d+)$',lab); ms=re.match(r'\s*(\d+(?:\.\d+)+)',sec)
    if m and ms: return f'x{ms[1]}-{m[1]}'
    m=re.search(r'종합문제\s*(\d+)',lab)
    if m: return f'z{int(m[1])}'
    return None
# SEMI: 앞쪽(문제편) 문항에 뒤쪽(풀이편, 문제 재수록+풀이) 병합
semi=[i for i in items if i['book']=='SEMI' and i.get('stem')]
first={}
for i in sorted(semi,key=lambda x:int(x.get('page') or 0)):
    k=semikey(i)
    if not k: continue
    if k in first and int(i.get('page') or 0)>=128:
        p0=first[k]
        if not p0.get('answer'): p0['answer']=i.get('answer')
        if i.get('solution_summary'): p0['solution_summary']=i['solution_summary']
        p0.setdefault('solution_ref',[]).append(i['id']); i['role']='해설'; i['merged_into']=p0['id']; i['stem']=None
    elif k not in first: first[k]=i
byk=collections.defaultdict(list)
for it in items:
    if it.get('role')!='해설' and it.get('stem'): byk[(it['book'],key(it))].append(it)
merged=0; orphan=[]
for it in items:
    if it.get('role')!='해설' or it.get('merged_into'): continue
    tb=it.get('solves_book') or it['book']; k=key(it)
    tgt=byk.get((tb,k)) if k else None
    if tgt and len(tgt)==1:
        p=tgt[0]
        if not p.get('answer') and it.get('answer'): p['answer']=it['answer']
        if it.get('solution_summary'):
            p['solution_summary']=(p.get('solution_summary') or '')+('' if not p.get('solution_summary') else '\n')+it['solution_summary']
        p.setdefault('solution_ref',[]).append(it['id']); it['merged_into']=p['id']; merged+=1
    else: orphan.append(it)
RF=yaml.safe_load(open(f'{R}/tools/review_flags.yaml')) or {}
for i in items:
    if i.get('id') in RF: i.setdefault('flags',[]); i['flags']=(i['flags'] or [])+[RF[i['id']]]
dup=[k for k,v in collections.Counter(i.get('id') for i in items).items() if v>1]
os.makedirs(f'{R}/items',exist_ok=True)
for bk in sorted({i['book'] for i in items}):
    L=[{k:v for k,v in i.items() if k!='_file'} for i in items if i['book']==bk]
    for i in L: i['book_title'],i['book_kind']=BOOKS.get(bk,(bk,''))
    yaml.safe_dump(L,open(f'{R}/items/{bk}.yaml','w'),allow_unicode=True,sort_keys=False,width=1000)
with open(f'{R}/all.jsonl','w') as w:
    for i in items: w.write(json.dumps({k:v for k,v in i.items() if k!='_file'},ensure_ascii=False)+'\n')
c=collections.Counter((i['book'],'해설' if i.get('role')=='해설' else '문항') for i in items)
rows=[f"| {b} | {BOOKS.get(b,(b,''))[0]} ({BOOKS.get(b,('',''))[1]}) | {c[(b,'문항')]} | {c[(b,'해설')]} |" for b in sorted({i['book'] for i in items})]
nq=sum(v for (b,r),v in c.items() if r=='문항'); na=sum(1 for i in items if i.get('role')!='해설' and i.get('answer'))
open(f'{R}/stats.md','w').write("# 교재 DB 현황\n\n| 코드 | 교재 | 문항 | 해설 |\n|---|---|---|---|\n"+'\n'.join(rows)+
  f"\n\n문항 {nq} · 답 있는 문항 {na} · 해설 병합 {merged} · 미병합 해설 {len(orphan)} · 중복 ID {len(dup)}\n")
print(f'items {len(items)} 문항 {nq} 답 {na} 병합 {merged} 미병합 {len(orphan)} 중복ID {dup[:5]}')
