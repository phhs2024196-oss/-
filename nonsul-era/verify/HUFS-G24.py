#!/usr/bin/env python3
"""HUFS-G24 (한국외대 2024학년도 논술전형 자연계 1~7) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HUFS-G24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str = "1") -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HUFS/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(가)(나)(다)로 a_1..a_20 을 차례로 생성해 S_20 직접 합산."""
    S = {1: 1, 2: 3}                       # S_1=a1, S_2=a1+a2
    for k in range(1, 10):                 # S_{2k+1}, S_{2k+2}
        S[2 * k + 1] = S[2 * k] + 3 * k + 1
        S[2 * k + 2] = S[2 * k + 1] + 2**k
    ans = sp.sympify(md_answer(1))
    alt = 1 + 2 + sum(3 * k + 1 for k in range(1, 10)) + sum(2**k for k in range(1, 10))
    report(1, S[20] == ans == alt, f"S_20={S[20]} (Σ식 {alt}), md={ans}")


def p02():
    """f=x^2+ax+b(최고차항 1의 이차식) 가정 → (나) 극한 조건으로 a,b 결정, (가) |ax+b|<=2 확인."""
    a, b = sp.symbols("a b", real=True)
    # (가) ⇒ f(x)/x^2 → 1 : 차수 3 이상·이차항 계수≠1 이면 |f-x^2| 이 유계가 아님
    f = x**2 + a * x + b
    sols = sp.solve([f.subs(x, 1), sp.limit((x**2 + a * x + b).subs(b, -1 - a) / (3 * x**2 - x - 2), x, 1) - sp.Rational(2, 5)], [a, b], dict=True)
    ok = len(sols) == 1
    fs = f.subs(sols[0])
    bounded = sp.expand(fs - x**2)               # 상수여야 (가) 만족
    okA = bounded.free_symbols == set() and abs(bounded) <= 2
    val = fs.subs(x, 2)
    ans = sp.sympify(md_answer(2))
    report(2, ok and okA and val == ans, f"f={fs}, f-x^2={bounded}, f(2)={val}, md={ans}")


def p03():
    """sin u = (-1+√5)/2, u=t^2-1 ∈[-1,35] 의 해 u 를 세고 각 u>-1 마다 t=±√(u+1) 2개."""
    s = (-1 + sp.sqrt(5)) / 2
    al = sp.asin(s)
    roots_q = sp.solve(x**2 + x - 1, x)
    ok_root = all((r == s) or (r < -1) for r in roots_q)     # 다른 근은 [-1,1] 밖
    us = []
    for k in range(-3, 10):
        for u in (al + 2 * k * sp.pi, sp.pi - al + 2 * k * sp.pi):
            if -1 <= float(u) <= 35:
                us.append(u)
    cnt = sum(2 if float(u) > -1 else 1 for u in us)
    # 수치 교차 확인: t 격자에서 부호변화 수
    import mpmath as mp
    F = lambda tt: mp.sin(tt**2 - 1) - (mp.sqrt(5) - 1) / 2
    N = 400000
    grid = [-6 + 12 * i / N for i in range(N + 1)]
    sc = sum(1 for i in range(N) if F(grid[i]) * F(grid[i + 1]) < 0)
    ans = sp.sympify(md_answer(3))
    report(3, ok_root and cnt == sc == ans, f"u 해 {len(us)}개 → t {cnt}개, 격자 부호변화 {sc}, md={ans}")


def p04():
    """f=12x(x-2)(x-α): (다)(라) ⟺ 0≤α≤2 를 격자 확인, S(α)=∫_0^2|f| 를 구간별 적분 후 최소화."""
    al = sp.symbols("alpha", real=True)
    f = 12 * x * (x - 2) * (x - al)
    def cond(av):
        xs = [-sp.Rational(k, 100) for k in range(1, 301, 7)] + [2 + sp.Rational(k, 100) for k in range(1, 301, 7)]
        return all((f.subs({al: av, x: xx}) < 0) if xx < 0 else (f.subs({al: av, x: xx}) > 0) for xx in xs)
    okc = all(cond(sp.Rational(k, 4)) for k in range(0, 9)) and not cond(sp.Rational(-1, 2)) and not cond(sp.Rational(5, 2))
    Smid = sp.expand(sp.integrate(f, (x, 0, al)) - sp.integrate(f, (x, al, 2)))
    S0 = sp.integrate(-f.subs(al, 0), (x, 0, 2))
    S2 = sp.integrate(f.subs(al, 2), (x, 0, 2))
    crit = [c for c in sp.solve(sp.diff(Smid, al), al) if c.is_real and 0 < c < 2]
    m = min([Smid.subs(al, c) for c in crit] + [S0, S2])
    # 수치 확인: α 격자에서 ∫|f|
    import mpmath as mp
    num = min(mp.quad(lambda xx: abs(12 * xx * (xx - 2) * (xx - k / 200)), [0, k / 200, 2]) for k in range(0, 401))
    ans = sp.sympify(md_answer(4))
    report(4, okc and m == ans and abs(num - m) < 1e-6, f"S(α)={Smid}, 임계점 {crit}, S(0)=S(2)={S0}, 최솟값 {m} (격자 {mp.nstr(num, 10)}), md={ans}")


def p05():
    """t=log_x y: (나)의 우변은 [2, 3-2^-100] 의 유리수 → k=3(t-2/t) 자연수 후보 열거, t 유리성 판정,
    t-2 와 1/t 의 이진 전개(100자리 이내)로 a_n, b_n 결정, 실제 x,y 존재 확인."""
    T = sp.symbols("T", positive=True)
    lo, hi = sp.Integer(2), 3 - sp.Rational(1, 2**100)
    kmin = sp.ceiling(3 * (lo - 2 / lo)); kmax = sp.floor(3 * (hi - 2 / hi))
    cands = []
    for k in range(int(max(kmin, 1)), int(kmax) + 1):
        for r in sp.solve(3 * T**2 - k * T - 6, T):
            if r.is_rational and lo <= r <= hi:
                cands.append((k, r))
    ok = len(cands) == 1
    k0, t0 = cands[0]
    def bits(q):                                  # 0<=q<1, 분모 2^m(m<=100) 인 이진 전개
        out = []
        for _ in range(100):
            q *= 2; d = 1 if bool(q >= 1) else 0; out.append(d); q -= d
        return out, q == 0
    A, okA = bits(t0 - 2)
    B, okB = bits(1 / t0)
    tot = sum(A) + sum(B)
    xy = next(((xx, sp.Integer(xx)**t0) for xx in range(2, 20) if (sp.Integer(xx)**t0).is_integer), None)
    okx = xy is not None and (3 * (sp.log(xy[1], xy[0]) - sp.log(xy[0]**2, xy[1]))).equals(k0)
    ans = sp.sympify(md_answer(5))
    report(5, ok and okA and okB and okx and tot == ans,
           f"후보 k∈[{kmin},{kmax}] 중 유리수 t: {cands}, (x,y) 예 {xy}, Σ(a_n+b_n)={tot}, md={ans}")


def p06():
    """f_n 을 정의대로 두고 x=0 좌우극한·미분계수 좌우극한을 sympy limit 으로 판정, n=1..100 합산."""
    h = sp.symbols("h", positive=True)
    total = 0
    tags = {}
    for k in range(1, 101):
        fL = lambda X: X**k * (1 + (-X)) / (-X)          # x<0: |x|=-x
        fR = lambda X: X**k * (1 + X) / X                # x>0
        cl = sp.limit(fL(-h), h, 0); cr = sp.limit(fR(h), h, 0)
        if not (cl == cr == 0):
            v = -1
        else:
            dl = sp.limit((fL(-h) - 0) / (-h), h, 0); dr = sp.limit((fR(h) - 0) / h, h, 0)
            v = 1 if dl == dr else 0
        tags[k] = v; total += v
    ans = sp.sympify(md_answer(6))
    report(6, total == ans, f"<f_1>={tags[1]}, <f_2>={tags[2]}, <f_3..100>={set(tags[k] for k in range(3,101))}, 합={total}, md={ans}")


def p07():
    """(1) 원 (x-c)^2+(y-c)^2=r^2 과 y=x^2+2 의 교점 방정식 F(x)=0. x=-1/2 가 유일한 교점이면 중근(극소)이어야
    하나 F'(-1/2) 가 c 와 무관하게 0 이 아님을 기호 계산으로 확인 + (c, r^2=… ) 격자에서 다른 실근 존재 확인 [증명-수치확인].
    (2) a=0: 중심 O, 반지름 2. a≠0, a≠-1/2: 법선 l' 과 y=x 의 교점 Q, 반지름 PQ 인 원(공식 구성)이 곡선과 P 에서만
    만나는지 유리수 a 표본에서 실근으로 확인. a<-1/2 에서는 추가 교점이 생겨 공식 (2) 가 성립하지 않음 —
    YAML 에 '공식해설오류' 태그·ALT 기록이 있어야 PASS(공식 해설은 유지)."""
    c, r2, a = sp.symbols("c r2 a", real=True)
    F = sp.expand((x - c)**2 + (x**2 + 2 - c)**2 - r2)
    okF = sp.expand(F - (x**4 + (5 - 2 * c) * x**2 - 2 * c * x + (2 * c**2 - 4 * c + 4 - r2))) == 0
    dF = sp.simplify(sp.diff(F, x).subs(x, sp.Rational(-1, 2)))
    ok1 = dF.free_symbols == set() and dF != 0
    # 수치: x=-1/2 를 지나는 (가) 원들(r2 = F(-1/2)=0 이 되도록) 은 항상 다른 실근을 가짐
    bad1 = 0
    for k in range(-40, 41):
        cv = sp.Rational(k, 4)
        r2v = sp.solve(F.subs({c: cv, x: sp.Rational(-1, 2)}), r2)[0]
        roots = set(sp.Poly(F.subs({c: cv, r2: r2v}), x).real_roots())
        if roots == {sp.Rational(-1, 2)}:
            bad1 += 1
    # (2) 공식 구성(중심 Q=l'∩(y=x), 반지름 PQ)의 교점 수를 a 표본에서 확인.
    #     a>-1/2 에서는 교점 P 하나, a<-1/2 에서는 다른 교점 2개가 생김(공식 해설 결함 → YAML 에 기록했는지 확인).
    bad_hi = 0; extra_lo = 0; n_hi = n_lo = 0
    samples = [sp.Rational(p_, q_) for p_ in range(-6, 7) for q_ in (1, 2, 3, 5) if sp.Rational(p_, q_) != sp.Rational(-1, 2)]
    X = sp.symbols("X")
    for av in sorted(set(samples)):
        bv = av**2 + 2
        if av == 0:
            cv = sp.Integer(0); r2v = sp.Integer(4)
        else:
            cv = sp.solve(sp.Eq(X - bv, -(X - av) / (2 * av)), X)[0]
            r2v = (av - cv)**2 + (bv - cv)**2
        roots = set(sp.Poly(F.subs({c: cv, r2: r2v}), x).real_roots())
        if av > sp.Rational(-1, 2):
            n_hi += 1; bad_hi += roots != {av}
        else:
            n_lo += 1; extra_lo += len(roots - {av}) == 2
    # 일반 a: F = (x-a)^2 · q(x), q 의 판별식 부호
    A_ = sp.symbols("A_", real=True)
    cA = sp.solve(sp.Eq(X - (A_**2 + 2), -(X - A_) / (2 * A_)), X)[0]
    FA = sp.together(F.subs({c: cA, r2: (A_ - cA)**2 + (A_**2 + 2 - cA)**2}))
    q, rem = sp.div(sp.Poly(sp.numer(FA), x), sp.Poly((x - A_)**2, x))
    qa, qb, qc = q.all_coeffs()
    disc = sp.factor(qb**2 - 4 * qa * qc)
    import pathlib, yaml
    it = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "items/HUFS/PAST" / f"{PREFIX}-07.yaml").read_text(encoding="utf-8"))
    flagged = any("공식해설오류" in str(tg) for tg in it.get("tags", []))
    ans1, ans2 = md_answer(7, "1"), md_answer(7, "2")
    ok = (okF and ok1 and bad1 == 0 and rem.is_zero and bad_hi == 0 and extra_lo == n_lo and flagged
          and ans1 is None and ans2 is None)
    report(7, ok, f"[증명-수치확인] (1) F'(-1/2)={dF}(c 무관), 격자 81개 반례 {bad1} / (2) a>-1/2 표본 {n_hi}개 실패 {bad_hi}; "
                  f"a<-1/2 표본 {n_lo}개 중 추가 교점 {extra_lo}개(판별식 {disc}) → 공식 (2) 는 a<-1/2 에서 거짓, YAML 표시 {flagged}")


if __name__ == "__main__":
    for name in sorted((k for k in dict(globals()) if re.fullmatch(r"p\d+", k)), key=lambda s: int(s[1:])):
        try:
            globals()[name]()
        except FileNotFoundError:
            print(f"{PREFIX}-{int(name[1:]):0{WIDTH}d} SKIP  YAML 없음")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
