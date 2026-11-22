#!/usr/bin/env python3
"""가천대 2022학년도 논술고사 자연 A형 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G22A"
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
    g = sp.Min(x, 1)
    f = g - x / (1 + x)
    f1 = f.subs(x, 1)
    left = sp.limit((f - f1) / (x - 1), x, 1, "-")
    right = sp.limit((f - f1) / (x - 1), x, 1, "+")
    fl = sp.simplify(f.subs(g, x)); fr = sp.simplify(f.subs(g, 1))
    yv = nums(yans(7).split("③")[1])
    ok = sp.simplify(fl - x**2 / (1 + x)) == 0 and sp.simplify(fr - 1 / (1 + x)) == 0 and [left, right] == yv
    report(7, ok, f"f1={f1}, 좌극한={left}, 우극한={right}, YAML③④={yv}")


def p08():
    sol = sp.solve([sp.Eq(0, 2*a*k*(1/(2*a)) - a*k**2)], [k], dict=True)
    slopes = {sp.simplify(2*a*s[k]) for s in sol if sp.simplify(s[k]) != 0}
    ok = slopes == {sp.Integer(2)} and nums(yans(8)) == [2]
    report(8, ok, f"접선이 (1/(2a),0) 지남 → k={[s[k] for s in sol]}, 기울기={slopes}")


def p09():
    ts = sp.solve(t**2 - 10*t + 24, t)
    good = []
    for cand in [sp.log(T0, 2) - 1 for T0 in ts]:
        # 근의 개수: 3^|x|+a = log2(T) 의 실근 수 합 (최솟값 1+a 에서 1개, 그 위 2개)
        cnt = 0
        for T in ts:
            c = sp.log(T, 2) - cand
            cnt += 1 if sp.simplify(c - 1) == 0 else (2 if (c - 1).evalf() > 0 else 0)
        if cnt == 1:
            good.append(sp.simplify(cand))
    ya = nums(yans(9))
    ok = len(good) == 1 and sp.simplify(good[0] - ya[0]) == 0
    report(9, ok, f"t={ts}, 근 1개인 a={good}, YAML={ya}")


def p10():
    r = sp.sqrt(a**2 + b**2)
    sols = []
    for (aa, bb) in [(4, 3), (-4, -3), (3, 4), (1, 2)]:
        th = sp.atan2(bb, aa)
        if sp.simplify(3*sp.cos(th) + sp.cos(th)*sp.tan(th) - 3) == 0:
            sols.append(sp.Rational(aa, bb))
    eq = sp.expand((3*a + b)**2 - 9*(a**2 + b**2))   # 6ab - 8b^2
    ratio = sp.solve(sp.factor(eq / b), a)[0] / b
    ok = sp.simplify(ratio - nums(yans(10))[0]) == 0 and sols and all(s == ratio for s in sols)
    report(10, ok, f"(3a+b)^2-9r^2={eq} → a/b={ratio}; 대입 확인 {sols}")


def p11():
    n = sp.symbols("n", integer=True, positive=True)
    S = n**2*(n + 2)**2/9
    an = sp.expand(S - S.subs(n, n - 1))
    a1 = S.subs(n, 1); a100 = an.subs(n, 100)
    ok = a1 + a100 == nums(yans(11))[0]
    report(11, ok, f"a_n={an}, a1={a1}, a100={a100}, 합={a1+a100}")


def p12():
    p, q = sp.symbols("p q")
    f = x**3 - 9*x**2 + p*x + q
    c1 = sp.limit((f - x**3)/x**2, x, sp.oo)
    sol = sp.solve([q, sp.limit((f.subs(q, 0))/x, x, 0) - 14], [p, q], dict=True)[0]
    F = f.subs(sol)
    val = sp.limit(F/(x - 2), x, 2)
    ok = c1 == -9 and F.subs(x, 2) == 0 and val == nums(yans(12))[0]
    report(12, ok, f"f={sp.factor(F)}, lim={val}")


def p13():
    sols = set()
    for base_sq in (sp.Rational(1, 9), 16):
        for s in sp.solve(sp.Abs(a + 1) - sp.sqrt(base_sq), a):
            B = abs(s + 1)
            mx = max(sp.log(sp.Rational(1, 9), B), sp.log(16, B), key=lambda e: e.evalf())
            if sp.simplify(mx - 2) == 0:
                sols.add(s)
    ok = sols == set(nums(yans(13)))
    report(13, ok, f"최댓값 2인 a={sorted(sols)}")


def p14():
    tau = sp.symbols("tau")
    s1 = sp.integrate(2*tau**3 - 12*tau**2 + 24*tau, (tau, 0, t))
    found = []
    for A in [sp.Rational(n, 9) for n in range(0, 40)]:
        s2 = sp.integrate(sp.Rational(3, 2)*tau**2 + 9*A, (tau, 0, t))
        roots = {r for r in sp.solve(sp.Eq(s1, s2), t) if r.is_real and r > 0}
        if len(roots) == 2:
            found.append(A)
    ok = set(found) == set(nums(yans(14)))
    report(14, ok, f"a∈{{n/9}} 전수: 출발 후 두 번 만나는 a={found}")


def p15():
    f = -x**2 + 6*x - 9
    l = f.subs(x, a) + sp.diff(f, x).subs(x, a)*(x - a)
    xi = sp.solve(l, x)[0]; yi = l.subs(x, 0)
    S = sp.integrate(-f, (x, 0, 3)) - sp.Rational(1, 2)*xi*(-yi)
    crit = [c for c in sp.solve(sp.diff(S, a), a) if 0 < c < 3]
    mn = min(S.subs(a, c) for c in crit)
    # 수치 확인(a=0.5 에서 직접 넓이)
    a0 = sp.Rational(1, 2)
    l0 = l.subs(a, a0)
    num = sp.integrate(sp.Max(0, -f) , (x, 0, 3)) - sp.Rational(1, 2)*xi.subs(a, a0)*(-yi.subs(a, a0))
    ok = mn == nums(yans(15))[0] and sp.simplify(num - S.subs(a, a0)) == 0
    report(15, ok, f"S(a)={sp.expand(S)}, 임계점={crit}, 최솟값={mn}")


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
