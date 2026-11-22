#!/usr/bin/env python3
"""KNU 2022학년도 논술(AAT) 모의고사 자연계열Ⅰ 수학 (KNU-M22A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-M22A"
WIDTH = 2
RESULTS = []
x, c = sp.symbols("x c", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())


def load_answer(seq, i=0):
    return load_item(seq)["subparts"][i]["answer"]["value"]




def circle(tv):
    """t 에서 원 R 의 중심 (f,g) 를 원 위·법선 조건으로 직접 풀기(f>0, g>0)."""
    F, G = sp.symbols("F G")
    sols = sp.solve([(tv - F) ** 2 + (tv ** 2 - G) ** 2 - G ** 2, 2 * tv * (G - tv ** 2) + (F - tv)], [F, G], dict=True)
    sub = lambda e: [e.subs(tv, v) for v in (sp.Rational(1, 10), 1, 3)]
    return [(s[F], s[G]) for s in sols if all(v > 0 for v in sub(s[F]) + sub(s[G]))]


def p01():
    import mpmath as mp
    mp.mp.dps = 40
    msgs, ok = [], True
    t = sp.symbols("t", positive=True)
    sols = circle(t)
    ok &= len(sols) == 1
    fv, gv = sols[0]
    L = sp.limit(fv * gv / t ** 3, t, 0, "+")
    a1 = sp.sympify(load_answer(1, 0))
    ok &= sp.simplify(L - a1) == 0
    msgs.append(f"1-1 f={sp.simplify(fv)}, g={sp.simplify(gv)}, 극한={L} md={a1}")

    # 1-2: θ 를 기하로 직접 계산(Q=수직선 x=t 와 원의 다른 교점) 후 중앙차분
    def theta(tt):
        tt = mp.mpf(tt)
        ff, gg = [mp.mpf(str(sp.N(e.subs(t, sp.Float(str(tt), 45)), 45))) for e in (fv, gv)]
        dy = mp.sqrt(gg ** 2 - (tt - ff) ** 2)
        P, Q = (tt, tt ** 2), (tt, 2 * gg - tt ** 2)          # 같은 수직선 위의 두 교점 (y = g ± dy)
        assert abs(P[1] - (gg + dy)) < 1e-25 or abs(P[1] - (gg - dy)) < 1e-25
        v1, v2 = (P[0] - ff, P[1] - gg), (Q[0] - ff, Q[1] - gg)
        return mp.acos((v1[0] * v2[0] + v1[1] * v2[1]) / gg ** 2)
    G = lambda tt: 1 / mp.sin(theta(tt) / 2) ** 2
    h = mp.mpf(10) ** -12
    der = (G(1 + h) - G(1 - h)) / (2 * h)
    a2 = sp.sympify(load_answer(1, 1))
    ok &= abs(der - a2) < 1e-6
    msgs.append(f"1-2 수치 미분={mp.nstr(der, 10)} md={a2}")

    # 1-3: 경계(곡선 O→P, 짧은 호 P→A, x축 A→O)를 촘촘히 잡아 신발끈 공식으로 S(t), t→0+ 추세
    def S(tt):
        tt = mp.mpf(tt)
        ff, gg = [mp.mpf(str(sp.N(e.subs(t, sp.Float(str(tt), 45)), 45))) for e in (fv, gv)]
        phiP = mp.atan2(tt ** 2 - gg, tt - ff)
        phiA = -mp.pi / 2
        # 짧은 호: P 에서 A 까지 각도 차가 π 미만이 되도록 방향 선택
        dphi = phiA - phiP
        while dphi > mp.pi: dphi -= 2 * mp.pi
        while dphi < -mp.pi: dphi += 2 * mp.pi
        pts = [(tt * k / 400, (tt * k / 400) ** 2) for k in range(401)]
        pts += [(ff + gg * mp.cos(phiP + dphi * k / 400), gg + gg * mp.sin(phiP + dphi * k / 400)) for k in range(1, 401)]
        pts += [(ff * (1 - mp.mpf(k) / 10), 0) for k in range(1, 10)]
        return abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2
    r = [S(tt) / mp.mpf(tt) ** 3 for tt in ("0.02", "0.01", "0.005")]
    a3 = sp.sympify(load_answer(1, 2))
    rich = 2 * r[2] - r[1]          # 오차가 O(t) 이므로 리처드슨 외삽
    ok &= abs(rich - a3) < 5e-4 and abs(r[2] - a3) < abs(r[1] - a3) < abs(r[0] - a3)
    msgs.append(f"1-3 S/t^3 (t=.02,.01,.005)={[mp.nstr(v, 7) for v in r]}, 외삽={mp.nstr(rich, 7)} md={a3}")
    report(1, ok, "; ".join(msgs))


def p02():
    msgs, ok = [], True
    A, B = sp.symbols("A B", positive=True)
    cs = sp.symbols("c0:4")
    f = 3 * x ** 4 + sum(cs[i] * x ** i for i in range(4))
    # 2-1: g 의 x=5 에서 연속·미분가능 조건을 a,b>0 일반으로 풀기
    s21 = sp.solve([(-A * f + A + B - B * f).subs(x, 5), (-A * sp.diff(f, x) - B * sp.diff(f, x)).subs(x, 5)], [cs[0], cs[1]], dict=True)[0]
    f5, fp5 = sp.simplify(f.subs(s21).subs(x, 5)), sp.simplify(sp.diff(f.subs(s21), x).subs(x, 5))
    md = [sp.sympify(v) for v in load_answer(2, 0).split(";")]
    ok &= [f5, fp5] == md
    msgs.append(f"2-1 (f(5),f'(5))=({f5},{fp5}) md={md}")
    # 2-3: f'=12(x-5)(x-p)^2, f(5)=1, a=2,b=3 → {g(p),g(5)}={3,6} 로 p 결정 후 h(t) 수치 확인
    p = sp.symbols("p", real=True)
    F = sp.integrate(12 * (x - 5) * (x - p) ** 2, x)
    F = F - F.subs(x, 5) + 1
    cands = []
    for case, gp in (("p<5", -2 * F.subs(x, p) + 5), ("p>5", 3 * F.subs(x, p))):
        g5 = 3
        for pv in sp.solve(sp.Eq(gp, 6 if case == "p>5" else 3), p):
            if pv.is_real and ((case == "p<5" and pv < 5) or (case == "p>5" and pv > 5)):
                vals = sorted([gp.subs(p, pv), g5 if True else None])
                if vals == [3, 6]:
                    cands.append(pv)
    ok &= cands == [6]
    Ff = sp.expand(F.subs(p, cands[0]))
    import numpy as np
    gnum = sp.lambdify(x, sp.Piecewise((-2 * Ff + 5, x < 5), (3 * Ff, True)), "numpy")
    xs = np.linspace(-5, 15, 200001)
    gv = gnum(xs)
    mono = bool(np.all(np.diff(gv) >= -1e-12))
    roots = sorted(set(sp.nsimplify(r) for r in sp.solve(Ff - 2, x) if r.is_real))
    tot = sum(roots)
    a3 = sp.sympify(load_answer(2, 2))
    # 2-2 [증명-수치확인]: 이 f 에서 h(t) 를 직접 셈 — g(x)=t 이고 g'(x)≠0 인 점(부호가 바뀌는 교점) 개수
    def h(tv):
        d = gv - tv
        idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
        return len(idx)
    hs = {tv: h(tv) for tv in (0, 2, 3, 4.5, 6, 7, 100)}
    # t=3,6 은 정류값 — g-t 가 부호를 바꾸지 않는 접촉(삼중근)이어야 h=0: 정류점 근방에서 확인
    stat = [float(gnum(np.array([5.0]))[0]), float(gnum(np.array([6.0]))[0])]
    fp = sp.diff(Ff, x)
    nstat = sorted(set(round(float(r), 9) for r in sp.solve(fp, x)))
    ok &= mono and tot == a3 and nstat == [5.0, 6.0] and sorted(stat) == [3.0, 6.0] and all(hs[k] == 1 for k in (0, 2, 4.5, 7, 100))
    msgs.append(f"2-2 f'=0 의 근={nstat}(서로 다른 2개, 6 은 중근) 정류값={stat}, h(t)(t≠3,6)={hs} [증명-수치확인]")
    msgs.append(f"2-3 p={cands}, f={sp.factor(Ff - 2)}+2, g 증가={mono}, f=2 근={roots} 합={tot} md={a3}")
    report(2, ok, "; ".join(msgs))


def seq_a(N):
    a = [None, sp.Integer(1)]
    for n in range(1, N):
        a.append(a[n] + 2 if n % 2 == 1 else a[n] / 2)
    return a


def p03():
    msgs, ok = [], True
    a = seq_a(80)
    # 3-1 [증명-수치확인]
    ok31 = all(a[2 * n - 1] == sum(sp.Rational(1, 2) ** (k - 1) for k in range(1, n + 1)) for n in range(1, 40))
    ok &= ok31
    msgs.append(f"3-1 n≤39 일치={ok31} [증명-수치확인]")
    # 3-2: n=1..6 에서 b_n 을 넓이 조건으로 직접 풀어 c_n = b_n (a_{2n-1})^n / a_{2n} 과 비교
    nn = sp.symbols("n", positive=True, integer=True)
    cmd = sp.sympify(load_answer(3, 1), locals={"n": nn})
    good = True
    for n in range(1, 7):
        A, Bv = a[2 * n - 1], a[2 * n]
        al = sp.symbols("al", positive=True)
        bb = (Bv * (1 - al / A)) / al ** n                   # P=(al, 선분 위 y) 가 곡선 위 → b
        region = sp.integrate(bb * x ** n, (x, 0, al)) + sp.Rational(1, 2) * (A - al) * Bv * (1 - al / A)
        sol = [s for s in sp.solve(sp.Eq(region, A * Bv / 2 / (n + 1)), al) if s.is_real and 0 < s < A]
        if len(sol) != 1:
            good = False
            break
        cn = sp.simplify(bb.subs(al, sol[0]) * A ** n / Bv)
        good &= sp.simplify(cn - cmd.subs(nn, n)) == 0
    ok &= good
    msgs.append(f"3-2 n=1..6 넓이 조건 직접 풀이와 c_n 일치={good}")
    # 3-3: 실제 좌표(a_15..a_18)로 사각형 이등분 점 Q
    A8, A9, B9, B8 = (a[15], 0), (a[17], 0), (0, a[18]), (0, a[16])
    al9 = [s for s in sp.solve(sp.Eq(sp.integrate((a[18] * (1 - sp.Symbol("u") / a[17]) / sp.Symbol("u") ** 9) * x ** 9, (x, 0, sp.Symbol("u")))
                                     + sp.Rational(1, 2) * (a[17] - sp.Symbol("u")) * a[18] * (1 - sp.Symbol("u") / a[17]),
                                     a[17] * a[18] / 2 / 10), sp.Symbol("u")) if s.is_real and 0 < s < a[17]][0]
    P9 = (al9, a[18] * (1 - al9 / a[17]))
    q = sp.symbols("q")
    Q = (q, a[16] * (1 - q / a[15]))
    ar = lambda P: sp.Rational(1, 2) * sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))
    qv = sp.solve(sp.Eq(sp.Abs(ar([A8, A9, P9, Q])) * 0 + ar([A8, A9, P9, Q]), ar([A8, A9, B9, B8]) / 2), q)
    qv = [v for v in qv if 0 <= v <= a[15]]
    a33 = sp.sympify(load_answer(3, 2))
    ok &= qv == [a33]
    msgs.append(f"3-3 P9={P9}, q={qv} md={a33}")
    report(3, ok, "; ".join(msgs))


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
