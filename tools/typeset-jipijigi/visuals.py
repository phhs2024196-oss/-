import re
def grade_chart():
    # 1~5등급 축, 값은 원고 표 1 그대로
    def x(g): return (g-1)/4*100
    return f'''<div class="viz gscale">
<div class="axis">{''.join(f'<span style="left:{x(g)}%">{g}등급</span>' for g in [1,2,3,4,5])}</div>
<div class="track"></div>
<div class="gap" style="left:{x(2.50)}%;width:{x(4.15)-x(2.50)}%"><b>평균 1.65등급 차이</b></div>
<div class="pt kyo" style="left:{x(2.50)}%"><i></i><em>교과전형<br><b>2.50등급</b></em></div>
<div class="pt non" style="left:{x(4.15)}%"><i></i><em>논술전형<br><b>4.15등급</b></em></div>
<p class="vnote">23개 대학 평균(2026학년도 자연계, 9등급제 기준). 학생부종합은 20개교, 논술과의 차이 평균 1.0등급.</p>
</div>'''
def bars(rows, total, unit='곳'):
    out=''.join(f'<div class="brow"><span class="bl">{l}</span><span class="bt"><span class="bf" style="width:{v/total*100:.1f}%"></span></span><span class="bv">{v}{unit}</span></div>' for l,v in rows)
    return f'<div class="viz bars">{out}<p class="vnote">막대 길이 = {total}개교 중 비율</p></div>'
def stats(items):
    return '<div class="viz stats">'+''.join(f'<div class="st"><b>{a}</b><span>{b}</span></div>' for a,b in items)+'</div>'
def chevrons(steps):
    return '<div class="viz chev">'+''.join(f'<span>{s}</span>' for s in steps)+'</div>'
def split(a,al,b,bl,cap):
    t=a+b
    return f'<div class="viz split"><div class="sb"><span class="sa" style="width:{a/t*100:.1f}%">{al} {a}곳</span><span class="sbb" style="width:{b/t*100:.1f}%">{bl} {b}곳</span></div><p class="vnote">{cap}</p></div>'

def apply(n, h):
    if n == 4:
        h = re.sub(r'<table>.*?</table>', grade_chart(), h, count=1, flags=re.S)
    if n == 6:
        h = h.replace('<blockquote class="remember">', stats([('약 20 : 1','경쟁률 중간값<br>(29개 대학)'),('41~66%','최저 충족 비율<br>(5개 대학)'),('88 → 24','겉보기 → 실질 경쟁률<br>(예시 1곳)')]) + '<blockquote class="remember">', 1)
    if n == 8:
        rows=[('교과 개념을 정확히 이해했는가',40),('문제를 풀 전략을 세울 수 있는가',21),('논리적으로 추론하고 정당화하는가',19),('계산을 정확히 하는가',19),('조건과 제시문을 정확히 읽는가',13),('과정을 수학적으로 표현·서술하는가',13)]
        h = re.sub(r'<table>.*?</table>', bars(rows,43), h, count=1, flags=re.S)
    if n == 20:
        h = h.replace('<strong>개념 → 생각하는 법 → 학교 색깔 → 실전.</strong>', '')
        h = h.replace('<p>책은 참고예요. 중요한 건 순서예요. </p>', '<p>책은 참고예요. 중요한 건 순서예요.</p>' + chevrons(['개념','생각하는 법','학교 색깔','실전']))
    if '수능 전 8곳, 수능 후 32곳' in h:
        h = re.sub(r'(수능 전 8곳, 수능 후 32곳.*?</ul>)', lambda m: m.group(1) + split(8,'수능 전',32,'수능 후','2027 자연계 시험일 확인 40곳'), h, count=1, flags=re.S)
    return h

# ===== v2 (지침 Q6: 결론형 제목, 빨강 1색 + 회색) =====
def fig(title, inner, note=''):
    n = f'<p class="vnote">{note}</p>' if note else ''
    return f'<div class="viz fig"><p class="ftitle">{title}</p>{inner}{n}</div>'

def stack_rows(rows, mx, legend):
    # rows: (label, [v1, v2], total_label)
    out = ''
    for l, vs, tl in rows:
        segs = ''.join(f'<span class="sg s{i}" style="width:{v/mx*100:.1f}%"></span>' for i, v in enumerate(vs) if v)
        out += f'<div class="brow"><span class="bl">{l}</span><span class="bt">{segs}</span><span class="bv">{tl}</span></div>'
    lg = ''.join(f'<span><i class="s{i}"></i>{t}</span>' for i, t in enumerate(legend))
    return f'<div class="legend">{lg}</div><div class="bars">{out}</div>'

def stack100(parts):
    t = sum(v for _, v, _ in parts)
    segs = ''.join(f'<span class="{c}" style="width:{v/t*100:.2f}%">{l} {v}곳</span>' for l, v, c in parts)
    return f'<div class="s100">{segs}</div>'

def dumbbell(rows, mx, lg=('겉보기 경쟁률','실질경쟁률')):
    out = ''
    for l, sub, a, b in rows:
        xa, xb = a/mx*100, b/mx*100
        out += (f'<div class="db"><span class="bl">{l}<small>{sub}</small></span><span class="dt">'
                f'<span class="dl" style="left:{xb}%;width:{xa-xb}%"></span>'
                f'<span class="dp g" style="left:{xa}%"></span><span class="dp r" style="left:{xb}%"></span>'
                f'<span class="dv g" style="left:{xa}%">{a:.2f}</span><span class="dv r" style="left:{xb}%">{b:.2f}</span></span></div>')
    axis = ''.join(f'<span style="left:{v/mx*100}%">{v}</span>' for v in range(0, mx+1, 20))
    return f'<div class="legend"><span><i class="s1"></i>{lg[0]}</span><span><i class="s0"></i>{lg[1]}</span></div>{out}<div class="dax"><span class="bl"></span><span class="dt">{axis}</span></div>'

def timeline(items):
    # items: (date, label, schools, highlight)
    cells = ''.join(f'<div class="tl{" hi" if h else ""}"><b>{d}</b><span class="tll">{l}</span><span class="tls">{s}</span></div>' for d, l, s, h in items)
    return f'<div class="tline">{cells}</div>'

def ranges(rows, mx, unit):
    out = ''
    for l, a, b, hi in rows:
        out += f'<div class="brow"><span class="bl">{l}</span><span class="bt"><span class="rg{" hi" if hi else ""}" style="left:{a/mx*100:.1f}%;width:{(b-a)/mx*100:.1f}%"></span></span><span class="bv">{a}~{b}{unit}</span></div>'
    return f'<div class="bars rng">{out}</div>'

def cap_table(h, key, repl):
    return re.sub(r'<p class="cap">\[표 ' + re.escape(key) + r'\][^<]*(?:<[^/][^>]*>[^<]*</[^>]+>[^<]*)*</p>\s*<table>.*?</table>', lambda m: repl, h, count=1, flags=re.S)

def before_cap(h, key, add):
    return re.sub(r'(<p class="cap">\[표 ' + re.escape(key) + r'\])', lambda m: add + m.group(1), h, count=1)

def apply_v2(n, h):
    h = h.replace('<div class="viz gscale">', '<p class="ftitle">같은 대학인데, 논술 합격자 내신이 교과보다 평균 1.65등급 낮았어요</p><div class="viz gscale">', 1)
    h = h.replace('<div class="viz bars">', '<p class="ftitle">43개교 중 40곳이 "개념을 제대로 이해했는가"를 봐요</p><div class="viz bars">', 1) if n == 8 else h
    if n == 12 and 'ftitle' not in h:
        r = ranges([('수능 수학', 3, 4, False), ('수리논술', 20, 60, True)], 60, '분')
        h = re.sub(r'(</p>)', lambda m: m.group(1) + fig('논술은 한 문제에 20~60분을 써요. 수능은 3~4분이에요', r, '문제당 평균 시간. 논술은 학교별 한 문제(소문항 포함) 기준.'), h, count=1)
    if '[표 6-2]' in h:
        rows = [('경북대', [90, 0], '90%'), ('건국대', [56, 6], '62%'), ('한국항공대', [25, 0], '25%'), ('고려대(세종)', [17, 6], '23%'), ('상명대', [0, 0], '분류 불가')]
        h = cap_table(h, '6-2', fig('경북대는 최근 3년 기출의 90%가 2028 범위 밖이에요', stack_rows(rows, 100, ['미적분Ⅱ가 필요한 문항', '기하가 필요한 문항']), '[그림 6-2] 최근 3개년 기출 중 2028 범위 밖 문항 비중. 상명대는 20문항 모두 단답형이라 범위를 가를 수 없었고, 고려대(세종)도 18문항 중 9문항이 그래요. 두 곳은 요강으로 다시 확인하세요.'))
    if '[표 6-4]' in h:
        h = before_cap(h, '6-4', fig('2028 논술전형 46곳 중 24곳은 학생부를 아예 안 봐요', stack100([('논술 100%', 24, 'a'), ('학생부 10~30%', 21, 'b'), ('자료 없음', 1, 'c')])))
    if '[표 8-1]' in h:
        rows = [('중앙대', '자연 · 대학 공개값', 88.40, 24.19), ('한국외대', '전체 · 대학 공개값', 51.12, 14.08), ('숭실대', '자연 · 최저 미충족·결시 제외', 37.98, 12.35), ('경희대', '자연 · 최저 충족률 반영', 35.53, 14.60), ('동국대', '자연 · 응시자 중 최저 충족자', 29.60, 10.62), ('홍익대', '자연 · 최저 충족 기준', 20.40, 6.47), ('한국항공대', '공학 · 결시·자격미달 제외', 19.72, 7.90), ('경북대', '자연 · 최저 통과 인원', 11.79, 4.53)]
        h = cap_table(h, '8-1', fig('최저를 맞추면 경쟁자는 겉보기의 27~41%로 줄었어요', dumbbell(rows, 100), '[그림 8-1] 겉보기 경쟁률과 실질경쟁률(2026). 산정 기준이 학교마다 달라 학교끼리 직접 비교하면 안 돼요.'))
    if '[표 7-3]' in h:
        items = [('11/19 (목)', '2027 수능', '', False), ('11/21 (토)', '수능 직후 주말', '고려대 · 건국대 · 서강대', True), ('11/22 (일)', '', '경희대 · 동국대 · 성균관대', True), ('11/28 (토)', '둘째 주말', '중앙대', False), ('11/29 (일)', '', '이화여대 · 한양대', False)]
        h = cap_table(h, '7-3', fig('10곳 중 9곳이 수능 뒤 두 주말에 시험을 봐요', timeline(items), '[그림 7-3] 주요 10개 대학 2027학년도 자연계 논술 시험일. 연세대는 2026 기준 9월 27일(수능 전).'))
    h = apply_8bu(h)
    return h

def hbars(rows, mx, unit, hi=0):
    out = ''.join(f'<div class="brow"><span class="bl">{l}</span><span class="bt"><span class="bf{" hi" if i < hi else ""}" style="width:{v/mx*100:.1f}%"></span></span><span class="bv">{v}{unit}</span></div>' for i, (l, v) in enumerate(rows))
    return f'<div class="bars hb">{out}</div>'

def after_table(h, circ, figure):
    m = re.search(r'<h3>(?:<span class="pm">@@\d+@@</span>)*' + circ, h)
    if not m: return h
    i = h.find('</table>', m.end())
    if i < 0: return h
    i += len('</table>')
    return h[:i] + figure + h[i:]

def apply_8bu(h):
    h = after_table(h, '①', fig('세종대·성신여대는 기출의 80% 이상이 미적분Ⅱ 내용이었어요', hbars([('세종대',83),('성신여대',80),('광운대',79),('아주대',75),('서강대',75),('이화여대',63)],100,'%',2), '최근 3개년 기출 중 미적분Ⅱ 내용이 필요한 문항 비중. 2028 범위는 아직 안 밝힌 학교들이에요.'))
    h = after_table(h, '③', fig('기하 비중은 한양대가 44%로 가장 높았어요', hbars([('한양대',44),('경희대',36),('동국대',27),('고려대',25),('중앙대',24),('연세대',22)],100,'%',1), '최근 3개년 기출 중 기하 단원이 포함된 문항 비중.'))
    h = after_table(h, '⑪', fig('추천 학교 난이도는 수준대마다 확실히 갈려요. 내 수준대부터 고르세요', ranges([('⑨ 최상위 추천 7곳',4.64,4.95,True),('⑩ 중상위 추천 6곳',4.0,4.33,False),('⑪ 중하위 추천 7곳',3.13,3.4,False)],6,''), '우리 기출 DB의 학교별 난이도 지표(43개교 평균 4.1). 막대는 추천 학교들의 최저~최고 값이에요.'))
    h = after_table(h, '⑬', fig('논술 100% 학교 8곳의 합격자 내신은 3.96~4.82등급이었어요', hbars([('서경대',4.82),('한신대',4.80),('덕성여대',4.76),('수원대',4.70),('동국대 (학생부 30%)',2.87)],9,'등급',4), '2026 논술 합격자 내신(9등급제). 내신을 공개한 8곳 중 값이 큰 4곳만 그렸어요. 동국대는 학생부를 반영해 비교용으로 넣었어요. 학교마다 지표(평균·50%컷)가 달라요.'))
    h = after_table(h, '⑭', fig('같은 학교인데 논술 경쟁률이 교과의 2.4~5.3배였어요', dumbbell([('숙명여대','자연',33.34,7.02),('건국대','자연',58.05,10.95),('홍익대','자연',20.40,7.20),('세종대','자연',27.12,11.28)],60,('논술 경쟁률','교과 경쟁률')), '2026 자연계 교과·논술 경쟁률(대 1). 예외로 수원대는 교과 14.63, 논술 13.08이었어요.'))
    return h
