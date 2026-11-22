#!/usr/bin/env python3
"""AJOU-M21 (아주대 2021학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "AJOU-M21"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _tri(P, Q, R):
    return sp.Abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2


def p01():
    k = sp.symbols("k", positive=True)
    X = sp.symbols("X", real=True)
    # (1)(2) 신발끈 공식으로 넓이를 직접 세우고 최대점을 반복
    xs, ss = [sp.Integer(1)], []
    for _ in range(8):
        xn = xs[-1]
        area = -sp.expand((xn * (k * X**2) - X * (k * xn**2)) / 2)   # 신발끈 공식, 0<X<xn 에서 양수
        xm = sp.solve(sp.diff(area, X), X)[0]
        ok_pos = sp.simplify(area.subs(X, xm)) > 0 and sp.diff(area, X, 2) < 0
        if not ok_pos:
            raise ValueError("최대 아님")
        xs.append(xm)
        ss.append(sp.simplify(area.subs(X, xm)))
    ok1 = sp.simplify(xs[1] - sp.sympify(md_answer(1, "1-1"))) == 0
    r = sp.simplify(ss[1] / ss[0])
    ok_geo = all(sp.simplify(ss[i + 1] / ss[i] - r) == 0 for i in range(len(ss) - 1))
    total = sp.simplify(ss[0] / (1 - r))
    ok2 = ok_geo and sp.simplify(total - sp.sympify(md_answer(1, "1-2"), locals={"k": k})) == 0
    # (3) 넓이 x(k-kx^2) 최대점 p, 정적분
    A = X * (k - k * X**2)
    p = [c for c in sp.solve(sp.diff(A, X), X) if c.is_positive][0]
    I = sp.integrate(X * (sp.sin(sp.pi * X**2) - sp.cos(sp.pi * X**2)), (X, 0, p))
    ok3 = sp.simplify(I - sp.sympify(md_answer(1, "1-3"))) == 0
    # 문제 1-2 (1)
    N = sp.symbols("N", positive=True)
    f = N * X**2
    roots = [rt for rt in sp.solve(sp.expand((1 - X) * sp.diff(f, X) - (f - f.subs(X, 1))), X) if rt != 1]
    P = (roots[0], f.subs(X, roots[0]))
    tn = sp.simplify(_tri((0, 0), (1, N), P))
    lim = sp.limit((tn + 1)**(1 / N), N, sp.oo)
    ok4 = len(roots) == 1 and lim == sp.sympify(md_answer(1, "2-1"))
    # (2) g(x)=(x^2+x-1)e^x-e 의 근이 x=1 뿐 [증명-수치확인]
    g = sp.lambdify(X, (X**2 + X - 1) * sp.exp(X) - sp.E)
    grid = [-60 + i * 0.001 for i in range(64001)]
    sign_changes = [v for a, v in zip(grid, grid[1:]) if g(a) * g(v) <= 0]
    ok5 = all(abs(v - 1) < 0.01 for v in sign_changes) and len(sign_changes) >= 1
    ok5 &= all(g(v) < 0 for v in grid if v < 0.99)
    # (3) f=kx^2+kx: k 격자에서 x≠1 실근 존재 여부
    K = sp.symbols("K", real=True)
    good = []
    for kv in [sp.Rational(i, 4) for i in range(-40, 41)]:
        if kv == 0:
            continue
        fk = kv * X**2 + kv * X
        eq = sp.expand((1 - X) * sp.diff(fk, X) - (fk - fk.subs(X, 1)))
        rts = [rt for rt in sp.solve(eq, X) if rt.is_real and rt != 1]
        if rts:
            good.append(kv)
    ans_set = sp.sympify(md_answer(1, "2-3"), locals={"Union": sp.Union, "Interval": sp.Interval, "oo": sp.oo})
    ok6 = len(good) == 80 and all(g_ in ans_set for g_ in good) and 0 not in ans_set
    report(1, all([ok1, ok2, ok3, ok4, ok5, ok6]),
           f"x1={xs[1]} Σs={total} ∫={sp.nsimplify(I)} t_n={tn} lim={lim} g 근 x=1 유일 [증명-수치확인] k≠0 전부 "
           f"({ok1},{ok2},{ok3},{ok4},{ok5},{ok6})")


Y, B, W = "Y", "B", "W"


def _color(covers):
    s = set(covers) - {W}
    if not s:
        return W if covers else None
    return "G" if s == {Y, B} else next(iter(s))


def p02():
    cols = (Y, B, W)
    pairs = list(itertools.product(cols, repeat=2))          # (안쪽, 바깥쪽)
    pY = Fraction(sum(1 for pr in pairs if _color(pr) == Y), 9)
    ok1 = pY == Fraction(md_answer(2, "1-1"))
    yel = [pr for pr in pairs if _color(pr) == Y]
    pk = Fraction(sum(1 for pr in yel if _color(pr[:1]) == Y), len(yel))
    ok2 = pk == Fraction(md_answer(2, "1-2"))
    blu = [pr for pr in pairs if _color(pr) == B]
    good = sum(1 for a in yel for b in blu
               if len({_color(a[:1]), _color(b[:1]), _color((a[1], b[1]))}) == 3)
    ok3 = Fraction(good, len(yel) * len(blu)) == Fraction(md_answer(2, "1-3"))

    def first_green(seq):
        for i in range(1, len(seq) + 1):
            if _color(seq[:i]) == "G":
                return i
        return None
    E = {kk: 0 for kk in range(2, 11)}
    for seq in itertools.product(cols, repeat=10):
        if seq[0] == Y:
            fg = first_green(seq)
            if fg is not None:
                E[fg] += 1
    total = sum(sp.log(E[kk], 10) for kk in E)
    c = sp.log(6, 10)
    ok4 = sp.simplify(sp.expand_log(total - sp.sympify(md_answer(2, "2-1"), locals={"c": c}), force=True)) == 0
    # (2) 전수: n 개 전구 × n 장, 조건 (가)(나) — 자료집 해석(나머지는 모두 흰색 커버, k개는 같은 순서)
    N = sp.symbols("n", integer=True, nonnegative=True)
    md = sp.sympify(md_answer(2, "2-2"), locals={"n": N})
    ok5 = True
    vals = {}
    for nv in range(1, 4):
        seqs = list(itertools.product(cols, repeat=nv))
        allw = tuple([W] * nv)
        Fk = [0] * (nv + 1)
        for assign in itertools.product(seqs, repeat=nv):
            nonw = [s for s in assign if s != allw]
            kk = len(nonw)
            if kk >= 1 and len(set(nonw)) == 1 and first_green(nonw[0]) == kk:
                Fk[kk] += 1
        vals[nv] = sum(Fk)
        ok5 &= vals[nv] == md.subs(N, nv)
    # 큰 n 은 공식형 독립 계산(직접 합)과 대조
    for nv in range(4, 15):
        direct = sum(math.comb(nv, kk) * (2**(kk - 1) - 1) * 3**(nv - kk) * 2 for kk in range(2, nv + 1))
        ok5 &= direct == md.subs(N, nv)
    report(2, all([ok1, ok2, ok3, ok4, ok5]),
           f"P(노)={pY} P(여전히 노)={pk} P(모두 다름)={Fraction(good, 9)} E_k={E} Σlog=36c? {ok4} ΣF(n=1..3)={vals}")


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
