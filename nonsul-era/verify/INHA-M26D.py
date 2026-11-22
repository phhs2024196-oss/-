#!/usr/bin/env python3
"""INHA-M26D (2026학년도 인하대 자연계열 논술 모의고사, 의예과) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML 정답과 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "INHA-M26D"
WIDTH = 2
RESULTS = []
KIT = pathlib.Path(__file__).resolve().parents[1]
x, t, s = sp.symbols("x t s", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int, sub: str) -> str:
    it = yaml.safe_load((KIT / f"items/INHA/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return next(p for p in it["subparts"] if str(p["id"]) == sub)["answer"]["value"]


def tex(v: str):
    """간단한 LaTeX(\\dfrac, \\sqrt, \\pi, \\ln) → sympy."""
    v = v.replace("\\dfrac", "\\frac")
    v = v.replace("\\pi", "{pi}")
    while "\\frac" in v:
        w = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", v)
        w = re.sub(r"\\frac(\w)(\w)", r"((\1)/(\2))", w)
        w = re.sub(r"\\frac(\w)\{([^{}]*)\}", r"((\1)/(\2))", w)
        w = re.sub(r"\\frac\{([^{}]*)\}(\w)", r"((\1)/(\2))", w)
        if w == v:
            raise ValueError(f"tex 변환 실패: {v}")
        v = w
    v = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", v)
    v = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", v)
    v = v.replace("\\pi", " pi ").replace("\\ln", " log ").replace("{", "(").replace("}", ")")
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    return parse_expr(v, local_dict={"pi": sp.pi, "log": sp.log, "sqrt": sp.sqrt},
                      transformations=standard_transformations + (implicit_multiplication_application, convert_xor))



from fractions import Fraction as Fr


def p02():
    """(2-1) 원점 접선 (2-2) g(t)=3 인 t (2-3) (a,k) 전수."""
    k, p, a, tt = sp.symbols("k p a tt", positive=True)
    q = -(x - 1) * (x - k)
    sol = sp.solve(sp.Eq(q.subs(x, p) - p * sp.diff(q, x).subs(x, p), 0), p)  # 접선이 원점 지남
    slope = sp.simplify(sp.diff(q, x).subs(x, [r for r in sol if r.is_positive is not False][0]))
    ok1 = sp.simplify(slope - (k + 1 - 2 * sp.sqrt(k))) == 0

    def count(av, kv, tv):
        n = 0
        for expr, lo, hi in ((-(x - 1) * (x - kv), 1, kv), (-av * (x - kv) * (x - 16), kv, 16)):
            rs = sp.Poly(sp.expand(expr - tv * x), x).nroots(n=30)
            rr = sorted({round(float(sp.re(r)), 9) for r in rs if abs(sp.im(r)) < 1e-12 and lo - 1e-12 <= sp.re(r) <= hi + 1e-12})
            n += len(rr)
        return n
    A, K = sp.Rational(1, 4), 3
    s1 = slope.subs(k, K)
    s2 = sp.simplify(A * (K + 16 - 8 * sp.sqrt(K)))                    # 오른쪽 조각 접선(대칭 계산)
    q2 = -A * (x - K) * (x - 16)
    sol2 = sp.solve(sp.Eq(q2.subs(x, p) - p * sp.diff(q2, x).subs(x, p), 0), p)
    s2b = sp.diff(q2, x).subs(x, [r for r in sol2 if r > 0][0])
    ok_s2 = sp.simplify(s2 - s2b) == 0
    cand = [c for c in (s1, s2b) if count(A, K, c) == 3]
    # 교점 수가 홀수가 되는 것은 접할 때뿐 → 후보 외 t 는 짝수 확인(격자)
    import numpy as np
    odd = [tv for tv in np.linspace(0.01, 3, 600) if count(A, K, sp.Float(tv)) % 2]
    ok2 = ok_s2 and len(cand) == 1 and sp.simplify(cand[0] - tex(ans(2, "2"))) == 0 and not odd
    # (2-3) k=2..15 전수: a 유리수 & 극한 조건
    found = []
    for kv in range(2, 16):
        av = sp.nsimplify(sp.radsimp((kv + 1 - 2 * sp.sqrt(kv)) / (kv + 16 - 8 * sp.sqrt(kv))))
        if not av.is_rational or av <= 0:
            continue
        sv = (kv + 1 - 2 * sp.sqrt(kv))
        e = sp.Rational(1, 10**4)
        if count(av, kv, sv + e) == 0 and count(av, kv, sv - e) == 4:
            found.append((av, kv))
    ok3 = found == [(sp.Rational(1, 4), 4), (sp.Integer(4), 9)]
    report(2, ok1 and ok2 and ok3, f"(2-1) 기울기 {slope} (2-2) t={cand} (2-3) {found}")


def gh(vals, xv):
    """꺾은선 f(정수점 값 vals) 에 대해 g(x), h(x) (Fraction)."""
    n = int(xv) if xv != int(xv) else int(xv)
    def f(z):
        i = min(int(z), len(vals) - 2)
        return vals[i] + (vals[i + 1] - vals[i]) * (z - i)
    pts = [Fr(j) for j in range(0, int(xv) + 1)] + [xv]
    fv = [f(z) for z in pts]
    M, m = max(fv), min(fv)
    g = max(z for z, v in zip(pts, fv) if v == M)
    h = max(z for z, v in zip(pts, fv) if v == m)
    return g, h


def p03():
    """(3-1) g=x,h=0 (3-2) 불연속점 (3-3) 5^8 기울기 전수 → 적분 최대·최소 꺾은선."""
    vals = [Fr(2 * j) for j in range(12)]
    ok1 = all(gh(vals, Fr(k, 7)) == (Fr(k, 7), 0) for k in range(1, 70))
    # (3-2) g 의 점프 위치: 격자 1/400 에서 |Δg| 가 큰 곳을 이분법으로
    v2 = [Fr(0)] + [Fr((-2) ** (n - 1)) for n in range(1, 12)]
    grid = [Fr(k, 400) for k in range(1, 4000)]
    jumps = []
    for a_, b_ in zip(grid, grid[1:]):
        if abs(gh(v2, b_)[0] - gh(v2, a_)[0]) > Fr(1, 10):
            lo, hi = a_, b_
            for _ in range(40):
                mid = (lo + hi) / 2
                if abs(gh(v2, mid)[0] - gh(v2, lo)[0]) > Fr(1, 10):
                    hi = mid
                else:
                    lo = mid
            jumps.append(round(float(hi), 6))
    ok2 = jumps == [2.5, 4.5, 6.5, 8.5]
    # (3-3) 전수(가지치기): 구간 n 의 조건은 n 까지의 f 로 결정
    def target(n):
        return Fr(0) if n % 2 else Fr((-1) ** (n // 2 + 1))
    sols = []
    def dfs(vals):
        n = len(vals) - 1
        if n == 8:
            sols.append(list(vals)); return
        for sl in (-2, -1, 0, 1, 2):
            nv = vals + [vals[-1] + sl]
            pts = [Fr(n) + Fr(k, 6) for k in (1, 2, 3, 4, 5)]
            d = [gh(nv + [nv[-1]], z) for z in pts]
            dd = [gg - hh for gg, hh in d]
            if all(dd[i + 1] - dd[i] == target(n + 1) * Fr(1, 6) for i in range(4)):
                dfs(nv)
    dfs([Fr(0)])
    integ = lambda v: sum((v[i] + v[i + 1]) / 2 for i in range(8))
    best = max(sols, key=integ); worst = min(sols, key=integ)
    uniq_max = sum(1 for s_ in sols if integ(s_) == integ(best)) == 1
    uniq_min = sum(1 for s_ in sols if integ(s_) == integ(worst)) == 1
    ok3 = (best == [0, 0, 2, 0, 0, 2, 2, 0, 0] and worst == [0, 0, 1, 0, -1, 1, 1, -1, -3]
           and uniq_max and uniq_min and integ(best) == 6 and integ(worst) == Fr(-1, 2))
    report(3, ok1 and ok2 and ok3, f"(3-1) g=x,h=0 (3-2) 점프 {jumps} (3-3) 조건 만족 {len(sols)}개, 최대 {[int(v) for v in best]}({integ(best)}), 최소 {[int(v) for v in worst]}({integ(worst)})")


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
