#!/usr/bin/env python3
"""가천대 2022학년도 논술고사 자연 B형 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G22B"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "GCU" / "PAST"
x, t, a, b, k, m = sp.symbols("x t a b k m", real=True)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def yans(seq):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return d["subparts"][0]["answer"]["value"]


def nums(s):
    """YAML answer 문자열에서 수(분수 포함)를 sympy 로 읽는다."""
    s = s.replace("\\log_2 3", "log(3,2)")
    return [sp.sympify(v) for v in re.findall(r"-?\d+/\d+|-?\d+|log\(3,2\)", s)]


def p07():
    # 모델: f = 1/(x-a)^2 꼴 발산, g = 3f - 2022 - (x-a) → 조건 만족. k 를 미지수로 풀이
    f = 1/x**2
    g = 3*f - 2022 - x
    limg = sp.limit(g, x, 0)
    r2 = sp.limit((3*f - g)/f, x, 0)
    r3 = sp.limit(g/f, x, 0)
    ks = sp.solve(sp.limit((k*f - 2*g)/(2*f - g), x, 0) + 9, k)
    ya = yans(7)
    ok = limg == sp.oo and r2 == 0 and r3 == 3 and ks == [15] and "15" in ya.split("④")[1] and "3" in ya.split("③")[1]
    report(7, ok, f"lim g={limg}, ②={r2}, ③={r3}, k={ks}")


def p08():
    f = x**2/2 - 2*x + 1
    g = 5 - 2*sp.Abs(x - 2)
    left = sp.solve(sp.Eq(f, 2*x + 1), x); right = sp.solve(sp.Eq(f, -2*x + 9), x)
    x1 = [r for r in left if r < 2]; x2 = [r for r in right if r >= 2]
    S = sp.integrate((2*x + 1) - f, (x, x1[0], 2)) + sp.integrate((-2*x + 9) - f, (x, 2, x2[0]))
    ya = yans(8)
    ok = x1 == [0] and x2 == [4] and S == sp.sympify(ya.split("④")[1].strip())
    report(8, ok, f"교점 {x1},{x2}, S={S}")


def p09():
    C = (3, 4); R = 5
    d = sp.Abs(3*C[0] + 4*C[1] + 10)/5
    lo, hi = d - R, d + R
    cnt = 0
    for n in range(1, 10):
        D = 2**n
        if D == lo or D == hi:
            cnt += 1
        elif lo < D < hi:
            cnt += 2
    ok = cnt == int(yans(9))
    report(9, ok, f"중심-직선 거리 {d}, {lo}≤D≤{hi}, 점 개수 {cnt}")


def p10():
    S = sp.Intersection(sp.solveset(x**2 - 2*x > 0, x, sp.S.Reals), sp.solveset(6 - x > 0, x, sp.S.Reals),
                        sp.solveset(x**2 - 2*x > 6 - x, x, sp.S.Reals))
    target = sp.Union(sp.Interval.open(-sp.oo, -2), sp.Interval.open(3, 6))
    ya = yans(10)
    ok = S == target and "x<-2" in ya.replace(" ", "") and "3<x<6" in ya.replace(" ", "")
    report(10, ok, f"해집합={S}")


def p11():
    r = sp.symbols("r", positive=True)
    S = sp.Rational(1, 2)*r*(80 - 2*r)
    rc = sp.solve(sp.diff(S, r), r)[0]
    mx = S.subs(r, rc)
    ok = mx == int(yans(11)) and 0 < (80 - 2*rc)/rc < 2*sp.pi
    report(11, ok, f"r={rc}, 중심각={(80-2*rc)/rc}, 최댓값={mx}")


def p12():
    f = x**3 - x + a; g = 2*x + b
    h = f - g
    vals = []
    for xc in sp.solve(sp.diff(h, x), x):
        dab = sp.solve(h.subs(x, xc), a)[0] - b   # a-b
        if dab < 0:
            vals.append(sp.simplify((f - g).subs(x, 1).subs(a, b + dab)))
    ok = vals == [int(yans(12))]
    report(12, ok, f"a<b 인 접점 경우 f(1)-g(1)={vals}")


def p13():
    lg2 = sp.Rational(3, 10)
    n = 1
    while sum((2*kk - 2)*lg2 for kk in range(1, n + 1)) <= 90:
        n += 1
    ok = n == int(yans(13))
    report(13, ok, f"log2=0.3 으로 Σlog a_k>90 인 최소 n={n} (n(n-1): {n*(n-1)}, {(n-1)*(n-2)})")


def p14():
    good = []
    AA = sp.Symbol("A")
    for A in sp.solve(sp.Eq(AA/2, 3/(2*AA + 4)), AA):
        den = x**2 + 2*A*x + 3
        bad = [r for r in sp.solve(den, x) if r.is_real and r <= 1]
        if not bad:
            good.append(A)
    ok = good == [int(yans(14))]
    report(14, ok, f"연속 후보 중 분모 0 없는 a={good}")


def p15():
    G = sp.Piecewise((-x**2/2 - 2*x, x < 0), (x**2/2 - 2*x, True))   # (|t|-2)의 부정적분, G(0)=0
    # f = G(x)-G(a); 두 실근 ⇔ G(a)=G(-2) 또는 G(2)
    sols = set()
    for c in (G.subs(x, -2), G.subs(x, 2)):
        for piece in ((-x**2/2 - 2*x, lambda v: v < 0), (x**2/2 - 2*x, lambda v: v >= 0)):
            for r in sp.solve(sp.Eq(piece[0], c), x):
                if piece[1](r):
                    sols.add(sp.nsimplify(r))
    ya = {sp.nsimplify(v) for v in [-2 - 2*sp.sqrt(2), -2, 2, 2 + 2*sp.sqrt(2)]}
    ystr = yans(15).replace(" ", "")
    ok = sols == ya and all(t_ in ystr for t_ in ("-2-2√2", "-2", "2", "2+2√2"))
    report(15, ok, f"G(a)=극값 인 a={sorted(sols, key=float)}")


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
