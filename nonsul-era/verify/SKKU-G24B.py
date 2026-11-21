#!/usr/bin/env python3
"""SKKU-G24B (성균관대 2024학년도 수시 논술우수전형 자연계 2교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "SKKU-G24B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    A = x**3 - 3 * x**2 + 2 * x + 8
    B = -x**4 + 8 * x**3 - 22 * x**2 + 25 * x + 13
    C = x
    D = -x**2 + 6 * x
    I = sp.Interval.open(0, sp.Rational(7, 2))
    # (i) [증명-수치확인] A-D >= 0 on (0,3.5): 최솟값 = 0 (임계점·끝점), 격자 확인
    h = A - D
    crit = [c for c in sp.solve(sp.diff(h, x), x) if c in I]
    hmin = min([h.subs(x, c) for c in crit] + [h.subs(x, 0), h.subs(x, sp.Rational(7, 2))])
    ok1 = hmin == 0 and all(h.subs(x, sp.Rational(i, 1000)) >= 0 for i in range(1, 3500)) \
        and (B - A).subs(x, 1) > 0
    # (ii) f(1)=8=A(1) → f-A 는 x=1 에서 최소(극소) → f'(1)=A'(1). 또한 A(1)<=8<=B(1) 확인
    ok_set = A.subs(x, 1) == 8 and B.subs(x, 1) >= 8
    f1 = sp.diff(A, x).subs(x, 1)
    ok2 = ok_set and f1 == sp.sympify(md_answer(1, "2"))
    # (iii) f(a)=g(a) ⇔ (g<=D<=A<=f 이므로) A(a)=D(a), a∈(0,3.5)
    sols = [c for c in sp.solve(sp.Eq(A, D), x) if c in I]
    a = sols[0]
    lim3 = 3 * sp.diff(A, x).subs(x, a) - 5 * sp.diff(D, x).subs(x, a)
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "3").replace(" ", "").split(";"))
    ok3 = len(sols) == 1 and a == sp.sympify(parts["a"]) and lim3 == sp.sympify(parts["limit"])
    # 수치 예: f=A+(x-2)^2*0 이 아닌 일반 예로 극한 확인 f=A+(x-2)^2/10, g=D-x(x-2)^2/20 (제시문 부등식 격자 확인)
    hh = sp.symbols("hh")
    fex = A + (x - 2)**2 / 10; gex = D - x * (x - 2)**2 / 20
    okex = all((B - fex).subs(x, sp.Rational(i, 100)) >= 0 and (gex - C).subs(x, sp.Rational(i, 100)) >= 0 for i in range(1, 350))
    limex = sp.limit((fex.subs(x, 2 + 3 * hh) - gex.subs(x, 2 + 5 * hh)) / hh, hh, 0)
    ok3 = ok3 and okex and limex == lim3
    # (iv) f(b)-g(b)=22, f-g <= B-C: B-C 가 (0,3.5)에서 22 가 되는 점이 최댓값 점인지
    pfun = B - C
    pts = [c for c in sp.solve(sp.Eq(pfun, 22), x) if c.is_real and c in I]
    crit = [c for c in sp.solve(sp.diff(pfun, x), x) if c in I]
    pmax = max(pfun.subs(x, c) for c in crit + [sp.Rational(7, 2)])
    res = set()
    for b in pts:
        assert pmax == 22
        res.add((b, 4 * sp.diff(B, x).subs(x, b) + 3 * sp.diff(C, x).subs(x, b)))
    md4 = {tuple(sp.sympify(v) for v in t) for t in sp.sympify(md_answer(1, "4"))}
    ok4 = res == md4
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인](i) min(A-D)={hmin}; f'(1)={f1}; a={a},lim={lim3}; (iv)={sorted(res)}")


def p02():
    # 실제 좌표로 직선-원 교점을 반복 계산(접선이면 같은 점). 각을 pi/12 격자에 스냅해 오차 누적 방지.
    def t(n):
        return 3 if n % 2 == 1 else 4

    def nxt(P, slope_ang):
        dx, dy = math.cos(slope_ang), math.sin(slope_ang)
        # |P + s d|^2 = 1, s≠0 근: s = -2 P·d
        s = -2 * (P[0] * dx + P[1] * dy)
        Q = (P[0] + s * dx, P[1] + s * dy)
        th = math.atan2(Q[1], Q[0])
        k = round(th / (math.pi / 12)) % 24
        return (math.cos(k * math.pi / 12), math.sin(k * math.pi / 12)), k

    P = [(-1.0, 0.0)]
    K = [12]
    for n in range(0, 2030):
        Q, k = nxt(P[-1], t(n + 1) * math.pi / 12)
        P.append(Q); K.append(k)
    # (i) 0<=n<=7 호 P_n P_{n+2}(짧은 쪽) 길이
    arcs = set()
    for n in range(0, 8):
        d = (K[n + 2] - K[n]) % 24
        arcs.add(min(d, 24 - d))
    arc_len = sp.pi / 12 * arcs.pop() if len(arcs) == 1 else None
    ok1 = arc_len is not None and sp.simplify(arc_len - sp.sympify(md_answer(2, "1"))) == 0
    # (ii) P_2024
    k = K[2024]
    exact = (sp.cos(k * sp.pi / 12), sp.sin(k * sp.pi / 12))
    md = sp.sympify(md_answer(2, "2"))
    ok2 = all(sp.simplify(a - b) == 0 for a, b in zip(exact, md))
    # (iii) A(n) 최댓값 (n=0..2025): 단위원 위 각 k_i*pi/12 의 세 점 삼각형 넓이(정확값)
    best = 0
    arg = []
    for n in range(0, 2026):
        pts = [(sp.cos(K[i] * sp.pi / 12), sp.sin(K[i] * sp.pi / 12)) for i in (n, n + 1, n + 2)]
        if len({K[n], K[n + 1], K[n + 2]}) < 3:
            continue
        ar = abs(sp.nsimplify(sp.N(((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1])) / 2, 30), [sp.sqrt(3)]))
        if n < 24:
            arg.append((n, float(ar)))
        if ar > best:
            best = ar
    ok3 = sp.simplify(best - sp.sympify(md_answer(2, "3"))) == 0
    nmax = [n for n, v in arg if abs(v - float(best)) < 1e-12]
    report(2, ok1 and ok2 and ok3, f"arc={arc_len} P2024=P_k(k*pi/12, k={k})={exact} maxA={best} at n={nmax}")


def p03():
    # (i) g=x^3+Ax^2+Bx+C, f=g' → p=3,q=2A,r=B. S_n=f(n) 등차수열의 합: a_1=S_1, a_n=S_n-S_{n-1} 공차 일정
    Aa, Bb, Cc = sp.symbols("A B C")
    g = x**3 + Aa * x**2 + Bb * x + Cc
    f = sp.diff(g, x)
    S = lambda n: f.subs(x, n)
    a = [S(1)] + [S(n) - S(n - 1) for n in range(2, 6)]
    eqs = [sp.expand(a[i + 1] - a[i] - (a[2] - a[1])) for i in range(4)]
    eqs = [e for e in eqs if e != 0]
    # h 미분가능(x=6 에서 좌미분 g'(6) = 우미분 h'(6+)=g'(0)), h(6)=h(0)+h(6) → g(0)=0
    eqs += [sp.diff(g, x).subs(x, 6) - sp.diff(g, x).subs(x, 0), g.subs(x, 0)]
    sol = sp.solve(eqs, [Aa, Bb, Cc], dict=True)[0]
    gs = g.subs(sol)
    h6 = gs.subs(x, 6)
    I1 = sp.integrate(gs, (x, 0, 6)) + sp.integrate(gs.subs(x, x - 6) + h6, (x, 6, 8))
    ok1 = I1 == sp.sympify(md_answer(3, "1"))
    # (ii) 정수 p,q,r(≠0) 전수: f 두 실근이 g=x^3+qx^2+px+r 의 근
    vals = set()
    for p in range(-15, 16):
        for q in range(-15, 16):
            for r in range(-15, 16):
                if 0 in (p, q, r) or q * q - 4 * p * r <= 0:
                    continue
                F = p * x**2 + q * x + r
                G = x**3 + q * x**2 + p * x + r
                if sp.rem(G, F, x) != 0:
                    continue
                al, be = sorted(sp.solve(F, x), key=lambda v: float(v))
                fx = F - G
                roots = sorted({al, be, *[c for c in sp.solve(fx, x) if c.is_real and al < c < be]}, key=float)
                val = sum(abs(sp.integrate(fx, (x, roots[i], roots[i + 1]))) for i in range(len(roots) - 1))
                vals.add(sp.nsimplify(sp.simplify(val)))
    ok2 = vals == {sp.sympify(md_answer(3, "2"))}
    # (iii) A,B 는 x^2+9C=0 의 두 근, C<0 정수 → 전수(A 정수)
    good = []
    for Av in range(-60, 61):
        for Cv in range(-500, 0):
            # A,B 가 x^2+9C=0 의 두 근: {A,B} = {±3sqrt(-C)}
            rt = 3 * math.sqrt(-Cv)
            if abs(abs(Av) - rt) > 1e-12:
                continue
            Bv = -Av
            G = x**3 + Av * x**2 + Bv * x + Cv
            cr = sp.solve(sp.diff(G, x), x)
            if len(cr) != 2 or not all(c.is_real for c in cr):
                continue
            c1, c2 = sorted(cr, key=float)          # 극대 c1, 극소 c2 (최고차 양수)
            mx, mn = G.subs(x, c1), G.subs(x, c2)
            if not (mx > 0 and mn < 0):
                continue
            if abs(float(mx - mn) - 24 * math.sqrt(6)) < 1e-9:
                good.append(Av)
    ok3 = good == [int(md_answer(3, "3"))]
    report(3, ok1 and ok2 and ok3, f"g={gs} 적분={I1}; (ii) 값={vals}; (iii) A={good}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
