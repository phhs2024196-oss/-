#!/usr/bin/env python3
"""SKKU-G22C (성균관대 2022학년도 수시 논술우수전형 자연계 3교시 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "SKKU-G22C"
WIDTH = 2
RESULTS = []
x, m, alpha = sp.symbols("x m alpha", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def rows(seq, nrows, rev_even):
    """수열을 삼각형으로 배열(k번째 줄 k개). rev_even 이면 짝수 줄 역순."""
    out, i = [], 0
    for k in range(1, nrows + 1):
        r = seq[i:i + k]
        i += k
        if rev_even and k % 2 == 0:
            r = r[::-1]
        out.append(r)
    return out


def build_F():
    """<제시문1> 조건으로 f, g, h 를 결정(정답 미사용)."""
    p = sp.symbols("p0:9")
    f = p[0] * x**2 + p[1] * x + p[2]
    g = p[3] * x**2 + p[4] * x + p[5]
    h = p[6] * x**2 + p[7] * x + p[8]
    eqs = [f.subs(x, 0), f.subs(x, 1), g.subs(x, 2), sp.diff(f, x, 2).subs(x, 0) + 2,
           (f - g).subs(x, 1), sp.diff(f - g, x).subs(x, 1),
           (g - h).subs(x, 2), sp.diff(g - h, x).subs(x, 2)]
    base = sp.solve(eqs, p[:8], dict=True)[0]
    hh = h.subs(base)
    # h 의 최댓값 2 (꼭짓점 > 2, 이차항 계수 < 0), f, g 의 최댓값은 2 미만
    cand = []
    for s8 in sp.solve(sp.Eq(hh.subs(x, -sp.diff(hh, x).coeff(x, 0) / sp.diff(hh, x).coeff(x, 1)), 2), p[8]):
        H = sp.expand(hh.subs(p[8], s8))
        lc = H.coeff(x, 2)
        if lc.is_real and lc < 0:
            vx = -H.coeff(x, 1) / (2 * lc)
            if vx > 2:
                cand.append(H)
    return sp.expand(f.subs(base)), sp.expand(g.subs(base)), cand


def p01():
    import numpy as np
    f, g, hs = build_F()
    ok1 = len(hs) == 1
    h = hs[0]
    p1 = kv(md_answer(1, "1"))
    ok1 &= all(sp.simplify(sp.sympify(p1[k], locals={"x": x}) - v) == 0 for k, v in (("f", f), ("g", g), ("h", h)))
    pieces = [(f, -sp.oo, 1), (g, 1, 2), (h, 2, sp.oo)]
    # (ii) (-1,0) 을 지나는 접선의 양수 접점
    t = sp.Symbol("t", real=True)
    tang = []
    for P, lo, hi in pieces:
        for r in sp.solve(sp.diff(P, x).subs(x, t) * (-1 - t) + P.subs(x, t), t):
            if r.is_real and r > 0 and (r > lo) and (r <= hi):
                tang.append(sp.nsimplify(r))
    md2 = sp.sympify(md_answer(1, "2"))
    ok2 = set(sp.simplify(v) for v in tang) == set(sp.simplify(v) for v in md2)
    # (iii) k(G(x)) 의 극댓값 개수 = 9 인 (a,b,c) 전수 — 촘촘한 격자에서 직접 셈
    fl = [sp.lambdify(x, P, "numpy") for P, _, _ in pieces]
    def Fn(u):
        return np.where(u <= 1, fl[0](u), np.where(u <= 2, fl[1](u), fl[2](u)))
    def k(u):
        return Fn(u) / (u + 1)
    cnt = 0
    for a, b, c in itertools.product(range(1, 7), repeat=3):
        X = np.arange(1, (2 * a + 2 * b + 2 * c) * 1000) / 1000.0
        G = np.where(X <= 2 * a, a - np.abs(X - a),
                     np.where(X <= 2 * a + 2 * b, b - np.abs(X - 2 * a - b), c - np.abs(X - 2 * a - 2 * b - c)))
        Y = k(G)
        nmax = int(np.sum((Y[1:-1] > Y[:-2]) & (Y[1:-1] > Y[2:])))
        cnt += nmax == 9
    ok3 = cnt == int(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"h={h} 접점={tang} 순서쌍={cnt}")


def p02():
    B = 40
    ok1 = True
    for N in range(2, 7):          # (i) f(0)<0, f(N)>0 동시 불가 [증명-수치확인]
        for a in range(-B, B + 1):
            if a == 0:
                continue
            for b in range(-B, B + 1):
                if (b * b - 1) % (4 * a):
                    continue
                c = (b * b - 1) // (4 * a)
                if c < 0 and a * N * N + b * N + c > 0:
                    ok1 = False
    # (ii) a<0, f(1/N)>0 인 모든 f
    md2 = md_answer(2, "2")
    expr = sp.sympify(md2.split(",")[0].split("=")[-1], locals={"x": x, "r": sp.Symbol("r")})
    ok2 = True
    for N in range(2, 7):
        found = set()
        for a in range(-B, 0):
            for b in range(-B, B + 1):
                if (b * b - 1) % (4 * a):
                    continue
                c = (b * b - 1) // (4 * a)
                if sp.Rational(a, N * N) + sp.Rational(b, N) + c > 0:
                    found.add((a, b, c))
        mdset = set()
        for r in range(1, N):
            P = sp.Poly(expr.subs(sp.Symbol("r"), r), x)
            mdset.add(tuple(int(P.coeff_monomial(x**i)) for i in (2, 1, 0)))
        ok2 &= found == mdset
    # (iii) f(0)<0, f(2/M)>0
    ok3 = True
    Ms = sp.Symbol("M")
    e3 = sp.sympify(md_answer(2, "3"), locals={"x": x, "M": Ms})
    for M in (3, 5, 7, 9):
        found = set()
        for a in range(-B, B + 1):
            if a == 0:
                continue
            for b in range(-B, B + 1):
                if (b * b - 1) % (4 * a):
                    continue
                c = (b * b - 1) // (4 * a)
                if c < 0 and a * sp.Rational(4, M * M) + b * sp.Rational(2, M) + c > 0:
                    found.add((a, b, c))
        P = sp.Poly(e3.subs(Ms, M), x)
        ok3 &= found == {tuple(int(P.coeff_monomial(x**i)) for i in (2, 1, 0))}
    # (iv) N=100, M=19 : Q, R 에서 a_n, b_n 을 차분으로 만들고 r 전수
    R = sp.Poly(e3.subs(Ms, 19), x).as_expr()
    vals = []
    for r in range(1, 100):
        Q = expr.subs(sp.Symbol("r"), r)
        an = [Q.subs(x, n) - (Q.subs(x, n - 1) if n > 1 else 0) for n in range(1, 11)]
        bn = [R.subs(x, n) - (R.subs(x, n - 1) if n > 1 else 0) for n in range(1, 11)]
        vals.append(sum(abs(p - q) for p, q in zip(an, bn)))
    p4 = kv(md_answer(2, "4"))
    ok4 = min(vals) == int(p4["min"]) and max(vals) == int(p4["max"])
    report(2, ok1 and ok2 and ok3 and ok4, f"(i) 반례없음={ok1} (ii)={ok2} (iii)={ok3} (iv) min={min(vals)} max={max(vals)} [증명-수치확인]")


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
