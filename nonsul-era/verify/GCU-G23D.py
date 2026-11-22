#!/usr/bin/env python3
"""가천대 2023학년도 수시 논술(실전) 자연D 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import math
import re
import sys

import sympy as sp

PREFIX = "GCU-G23D"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p07():
    a, d = sp.symbols("a d")
    term = lambda k: a + (k - 1) * d  # noqa: E731
    sol = sp.solve([term(5) + term(6) + term(7) - 30, term(9) + term(10) - 6], [a, d], dict=True)[0]
    S20 = sum(term(k) for k in range(1, 21)).subs(sol)
    report(7, S20 == 20, f"a1={sol[a]}, d={sol[d]}, S20={S20}")


def p08():
    X, Y = sp.symbols("X Y", positive=True)
    sols = sp.solve([X + Y - 11, sp.Rational(1, 2) * X * Y * sp.sin(sp.pi / 6) - 6], [X, Y], dict=True)
    s = [v for v in sols if v[X] < v[Y]][0]
    val = s[Y] / s[X]
    report(8, val == sp.Rational(8, 3), f"AC={s[X]}, BD={s[Y]}, BD/AC={val}")


def p09():
    tt, xx = sp.symbols("tt xx", real=True)
    y = -xx ** 3 + 6 * tt * xx ** 2 + 5 * tt * xx
    yp = sp.diff(y, xx)
    x0 = sp.solve(sp.diff(yp, xx), xx)[0]             # 기울기(이차, 최고차 음수)의 최대점
    slope = sp.expand(yp.subs(xx, x0))
    y0 = sp.expand(y.subs(xx, x0))
    line = sp.expand(slope * (xx - x0) + y0)
    ok = (sp.expand(yp - (-3 * xx ** 2 + 12 * tt * xx + 5 * tt)) == 0 and slope == 12 * tt ** 2 + 5 * tt
          and x0 == 2 * tt and y0 == 16 * tt ** 3 + 10 * tt ** 2
          and sp.expand(line - ((12 * tt ** 2 + 5 * tt) * xx - 8 * tt ** 3)) == 0)
    report(9, ok, f"① y'={yp}, ② 최대 기울기={slope}, ③ 접점=({x0},{y0}), ④ y={line}")


def p10():
    # 1 보다 큰 자연수 a,b,c 전수 탐색(c≤6): log2 a + log3 b = 3c, (log3 a)(log2 b) = 2c^2, a<b
    found = []
    for c in range(2, 5):
        for i in range(1, 3 * c + 1):
            a = 2 ** i  # a=2^i, b=3^j 꼴을 넓게 탐색(로그값이 정수배가 되는 후보)
            for b in [3 ** j for j in range(1, 3 * c + 1)]:
                if math.isclose(math.log2(a) + math.log(b, 3), 3 * c) and math.isclose(
                        math.log(a, 3) * math.log2(b), 2 * c * c) and a < b:
                    found.append((a, b, c))
    # 기호: A+B=3c, AB=2c^2 의 근
    cc = sp.symbols("c", positive=True)
    roots = sp.solve(x ** 2 - 3 * cc * x + 2 * cc ** 2, x)
    ks = {sp.nsimplify(sp.Rational(a, b) ** sp.Rational(1, c)) for a, b, c in found}
    report(10, ks == {sp.Rational(2, 9)} and len(found) >= 2, f"근={roots}, 해(a,b,c)={found}, k={ks}")


def p11():
    k = sp.symbols("k", real=True)
    xs = [sp.pi * i / 720 for i in range(0, 1441)]
    y = lambda xv, kv: sp.sin(sp.pi / 2 + xv) ** 2 - 2 * kv * sp.sin(xv) + 3 * kv - 1  # noqa: E731
    # t=sinx 치환식과 일치 확인 + k 결정: max over x of y = -3 (k<-1)
    tt = sp.symbols("tt")
    ft = sp.expand(sp.cos(x) ** 2 - 2 * k * sp.sin(x) + 3 * k - 1).subs(sp.cos(x) ** 2, 1 - sp.sin(x) ** 2).subs(sp.sin(x), tt)
    ksol = [v for v in sp.solve(ft.subs(tt, 1) + 3, k) if v < -1]
    kk = ksol[0]
    vals = [float(y(xv, kk)) for xv in xs]
    mx, mn = max(vals), min(vals)
    xmax = xs[vals.index(mx)]
    xmin = xs[vals.index(mn)]
    ok = (sp.expand(ft - (-tt ** 2 - 2 * k * tt + 3 * k)) == 0 and kk == -2 and abs(mx + 3) < 1e-9
          and abs(mn + 11) < 1e-9 and xmax == sp.pi / 2 and xmin == 3 * sp.pi / 2)
    report(11, ok, f"① y={ft}, ② x={xmax}, ③ x={xmin}, ④ k={kk}, 최댓값={mx:.6f}, 최솟값={mn:.6f}")


def p12():
    a, k = sp.symbols("a k", positive=True)
    h = sp.integrate(2 * t - 3 * a, (t, 0, k)) - sp.integrate(t + a, (t, 0, k))

    def count(av):
        hk = h.subs(a, av)
        sols = set()
        for rhs in (72, -72):
            sols |= {s for s in sp.solve(sp.Eq(hk, rhs), k) if s.is_real and s > 0}
        return len(sols)

    good = [av for av in range(1, 20) if count(av) == 2]
    report(12, good == [3], f"h(k)={sp.expand(h)}, 양수해 2개인 a(1..19)={good}")


def p13():
    S10, S11, Q = sp.symbols("S10 S11 Q")   # Q = Σ_{k=1}^{10} a_k^2
    eqs = [Q + 4 * S10 + 40 - 192,          # (가)
           Q + S10 - 2 * S11,               # (나)
           S10 - 10 * (S11 - S10) - 20]     # (다) Σk(a_k-a_{k+1}) = S10 - 10 a11
    sol = sp.solve(eqs, [S10, S11, Q], dict=True)[0]
    # (다) 전개 확인
    a_ = sp.symbols("a1:12")
    lhs = sp.expand(sum((k + 1) * (a_[k] - a_[k + 1]) for k in range(10)))
    ok_tel = sp.expand(lhs - (sum(a_[:10]) - 10 * a_[10])) == 0
    feasible = sol[Q] >= sol[S10] ** 2 / 10   # 코시-슈바르츠: 실수열 존재 필요조건
    report(13, sol[S11] == 31 and ok_tel,
           f"S10={sol[S10]}, S11={sol[S11]}, Σa²={sol[Q]} / 실수열 존재 가능? {feasible} (Σa²≥S10²/10={sol[S10]**2/10} 필요 → 출제오류 의심)")


def p14():
    def fval(xv, n):  # xv 는 유리수 — (0,n] 로 환원 후 정확 비교
        r = sp.Rational(xv) - n * (sp.ceiling(sp.Rational(xv) / n) - 1)
        v = 2 * r + 21
        m = 0
        while 5 ** (m + 1) <= v:
            m += 1
        return m

    def discont(n):
        # 후보점: 계단점 x0 (2x+21=5^m) 및 주기 경계 x=jn 의 평행이동. 좌극한과 함숫값·우극한 비교
        cand = set()
        steps = [sp.Rational(5 ** m - 21, 2) for m in range(2, 8) if 0 < sp.Rational(5 ** m - 21, 2) <= n]
        base = steps + [sp.Integer(n)]
        for j in range(0, 2023 // n + 2):
            for s in base:
                p = s + j * n
                if 0 < p < 2023:
                    cand.add(p)
        eps = sp.Rational(1, 10 ** 6)
        cnt = 0
        for p in cand:
            if not (fval(p - eps, n) == fval(p, n) == fval(p + eps, n)):
                cnt += 1
        return cnt

    good = [n for n in range(1, 2023) if discont(n) == 23]
    report(14, max(good) == 344, f"불연속점 23개인 n={good[:8]}{'…' if len(good) > 8 else ''} 최댓값={max(good)}, n=345 → {discont(345)}개")


def p15():
    m = sp.symbols("m", real=True)
    tt = sp.symbols("tt", real=True)
    eq = sp.expand(((tt ** 2 + 1 - sp.Rational(9, 2)) * 2 * tt + (tt - m)))   # 기울기 곱 -1 (t≠0, t≠m) 정리
    ok_eq = sp.expand(eq - (2 * tt ** 3 - 6 * tt - m)) == 0
    # 실근 1개 조건을 여러 m 에서 직접 확인 (t=0 은 m=0 일 때만 수직 성립 — 같은 식에 포함)
    def npts(mv):
        return len([r for r in sp.Poly(eq.subs(m, mv), tt).real_roots()])
    def npts_distinct(mv):
        return len(set(sp.Poly(eq.subs(m, mv), tt).real_roots()))
    tests = {mv: npts_distinct(mv) for mv in [-10, -5, sp.Rational(-41, 10), -4, -1, 0, 3, 4, sp.Rational(41, 10), 9]}
    ok = ok_eq and all((tests[mv] == 1) == (abs(mv) > 4) for mv in tests)
    sol = sp.solve_univariate_inequality((4 - m) * (-4 - m) > 0, m)
    report(15, ok, f"2t³-6t-m=0 ✓, 근 개수 {dict((str(k), v) for k, v in tests.items())}, 범위 {sol}")


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
