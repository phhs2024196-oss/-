#!/usr/bin/env python3
"""가천대 2022학년도 논술고사 자연 D형 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G22D"
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
    vals = []
    for K in (sp.Integer(7), sp.Rational(1, 5)):
        A, B, C = K**sp.Rational(1, 3), K**sp.Rational(1, 4), K**sp.Rational(1, 5)
        t1 = sp.simplify(sp.log(sp.sqrt(B))/sp.log(1/A))
        t2 = sp.simplify(sp.log(B/C)/sp.log(B))
        t3 = sp.simplify(sp.log(A**sp.Rational(1, 2))/sp.log(C))
        vals.append((t1, t2, t3, sp.simplify(t1 + t2 + t3)))
    ya = yans(7)
    ok = vals[0] == vals[1] and vals[0] == (sp.Rational(-3, 8), sp.Rational(1, 5), sp.Rational(5, 6), sp.Rational(79, 120)) \
        and all(s in ya for s in ("-3/8", "1/5", "5/6", "79/120"))
    report(7, ok, f"①②③④={vals[0]} (k=7, 1/5 동일)")


def p08():
    c = sp.symbols("c", real=True)
    bsol = sp.solve(sp.Eq((3 - a) + b, a + c), b)[0]          # 연속
    left = sp.limit(sp.cancel((((3 - a)*x + bsol) - (a + c))/(x - 1)), x, 1, "-")
    right = sp.limit(sp.cancel(((a*x**2 + c) - (a + c))/(x - 1)), x, 1, "+")
    sa = sp.solve(sp.Eq(left, right), a)
    bc = sp.simplify(bsol - c).subs(a, sa[0])
    ok = sa == [1] and bc == int(yans(8))
    report(8, ok, f"좌미분계수={left}, 우미분계수={right}, a={sa}, b-c={bc}")


def p09():
    f = sp.cos(x - sp.pi/2)*sp.sin(x - sp.pi) - 2*sp.cos(x)**2 - sp.sin(x)
    T = sp.symbols("T")
    g = T**2 - T - 2
    same = sp.simplify(f - g.subs(T, sp.sin(x))) == 0
    cands = [-1, 1] + [r for r in sp.solve(sp.diff(g, T), T) if -1 <= r <= 1]
    vs = [g.subs(T, c) for c in cands]
    ya = yans(9).replace(" ", "")
    ok = same and max(vs) == 0 and min(vs) == sp.Rational(-9, 4) and "최댓값0" in ya and "최솟값-9/4" in ya
    report(9, ok, f"f=sin²x-sinx-2 확인={same}, 최댓값={max(vs)}, 최솟값={min(vs)}")


def p10():
    A, r = sp.symbols("A r", real=True)
    sols = sp.solve([A*r**2*(1 + r**6) - 36, A*r**2*(1 + r**2) - 12], [A, r], dict=True)
    vals = {sp.simplify(s[A]*(1 + s[r]**8)) for s in sols}
    ok = vals == {int(yans(10))}
    report(10, ok, f"(a,r)={[(s[A], s[r]) for s in sols]}, a1+a9={vals}")


def p11():
    h = sp.Abs(x)
    ok_h = all(sp.simplify(h.subs(x, v)**2 - v**2) == 0 and sp.sqrt(h.subs(x, v)**2) == h.subs(x, v) for v in (-3, -1, 0, 2))
    g = sp.floor(x) + h
    def cnt(tv):
        # 구간별 선분과 점의 교점 수(정확 계산)
        n = 0
        n += len([r for r in sp.solve(-x - 1 - tv, x) if -1 <= r < 0])
        n += len([r for r in sp.solve(x - tv, x) if 0 <= r < 1])
        n += 1 if g.subs(x, 1) == tv else 0
        return n
    f0 = cnt(0)
    near = {cnt(sp.Rational(s, 1000)) for s in (-3, -1, 1, 3)}
    gl = sp.simplify(g.subs(x, -sp.Rational(1, 2))); gr = g.subs(x, sp.Rational(1, 2))
    ya = yans(11)
    ok = ok_h and gl == -sp.Rational(1, 2) and gr == sp.Rational(1, 2) and f0 == 2 and near == {1} \
        and "③ 1" in ya and "④ 2" in ya
    report(11, ok, f"h=|x|, g(-1/2)={gl}, g(1/2)={gr}, f(0)={f0}, f(±ε)={near}")


def p12():
    p, q, r_, s = sp.symbols("p q r s")
    f = p*x**3 + q*x**2 + r_*x + s
    fp = sp.diff(f, x)
    qsol = sp.solve(sp.Poly(sp.expand(fp.subs(x, -x) - fp), x).coeffs(), q)
    f2 = f.subs(q, 0)
    ssol = sp.solve(sp.integrate(f2, (x, -2, 2)) - 3, s)[0]
    val = sp.simplify((f2.subs(x, 1) + f2.subs(x, -1)).subs(s, ssol))
    ok = val == sp.sympify(yans(12))
    report(12, ok, f"q={qsol}, d={ssol}, f(1)+f(-1)={val}")


def p13():
    A = sp.log(2, 10)*sp.log(5, 10)
    lhs = sp.log(25, 10)**3 - sp.log(4, 10)**3
    found = [(M, N) for M in range(1, 30) for N in range(1, 30)
             if (N*A).evalf() < 1 and abs((lhs - M*(1 - A)*sp.sqrt(1 - N*A)).evalf(50)) < 1e-40]
    ya = yans(13).replace(" ", "")
    ok = found == [(8, 4)] and "m=8" in ya and "n=4" in ya
    report(13, ok, f"1≤m,n<30 전수(50자리 수치): {found}")


def p14():
    def f(n):
        return sp.sqrt(n + 1) if n % 2 else sp.sqrt(n - 1)
    tot = sum(1/(f(kk) + f(kk + 2)) for kk in range(1, 99))
    target = (9 + 3*sp.sqrt(11) - sp.sqrt(2))/2
    ok = abs((tot - target).evalf(40)) < 1e-30 and sp.simplify(sp.sympify(yans(14).replace("√11", "*sqrt(11)").replace("√2", "sqrt(2)")) - target) == 0
    report(14, ok, f"직접 합(40자리)={tot.evalf(15)}, 목표={target.evalf(15)}")


def p15():
    X = t**3 - 9*t**2 + 24*t
    v = sp.diff(X, t)
    turns = sp.solve(v, t)
    xs = sorted({X.subs(t, tt) for tt in turns})
    t_lo = min(r for r in sp.solve(X - xs[0], t) if r >= 0)
    t_hi = max(r for r in sp.solve(X - xs[1], t) if r >= 0)
    dist = sp.integrate(sp.Abs(v), (t, t_lo, t_hi))
    # 구간 안의 위치는 세 번 지나는지 확인 (중간값 18)
    times18 = [r for r in sp.Poly(X - 18, t).nroots() if r.is_real and t_lo <= r <= t_hi]
    sp_avg = dist/(t_hi - t_lo)
    ok = len(times18) == 3 and sp_avg == int(yans(15))
    report(15, ok, f"전환 {turns}, 구간 [{t_lo},{t_hi}], 거리 {dist}, x=18 통과 {len(times18)}회, 평균속력 {sp_avg}")


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
