import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("UOS-G22B", "items/UOS/PAST")
T = sp.Symbol("t")


def comps(total, parts):
    if parts == 1:
        yield (total,); return
    for i in range(total + 1):
        for r in comps(total - i, parts - 1):
            yield (i,) + r


def p01():
    cnt = 0; by = {1: 0, 2: 0, 3: 0}
    for s in comps(5, 4):
        for t in comps(5, 4):
            if all(not (s[i] > 0 and t[i] > 0) for i in range(4)):
                cnt += 1; by[sum(1 for v in s if v)] = by.get(sum(1 for v in s if v), 0) + 1
    V.check(1, "1-1 전수", cnt, V.ans(1, "1-1"))
    V.report(1, (by[1], by[2], by[3]) == (84, 144, 24), f"종류수별 {by[1]},{by[2]},{by[3]}")


def p02():
    mp.mp.dps = 40
    f = lambda x: x**3 - 2*x + 11 if x <= -2 else mp.mpf(5)/2*x - 2*mp.cos(mp.pi/3*x) + 11
    # 연속·증가 확인(역함수 존재) 및 교점: f(v)=5v+5 의 모든 근을 촘촘한 격자에서 부호변화로 탐색
    xs = [mp.mpf(i)/50 for i in range(-1500, 1500)]
    vals = [f(x) for x in xs]
    inc = all(b > a for a, b in zip(vals, vals[1:]))
    roots = []
    g = lambda x: f(x) - 5*x - 5
    for a_, b_ in zip(xs, xs[1:]):
        if g(a_) == 0: roots.append(a_)
        elif g(a_) * g(b_) < 0: roots.append(mp.findroot(g, (a_, b_), solver="anderson"))
    a = sum(roots)
    V.check(2, "2-1 정수부분", sp.Integer(int(mp.floor(a))), V.ans(2, "2-1"))
    V.report(2, inc and len(roots) == 2 and abs(roots[0] + 3) < 1e-20 and 3 < roots[1] + 0 < 4,
             f"f 증가={inc}, 근 {[mp.nstr(r, 8) for r in roots]}, a={mp.nstr(a, 8)}")


def p03():
    # 직접 개수: k^2 m^3 = 2^(9n) 의 모든 자연수 해 중 m<=8^n<=m^2 (k,m 은 2의 거듭제곱임을 소인수분해로 확인)
    def a_n(n):
        c = 0
        for s in range(0, 9*n//2 + 1):
            for t in range(0, 3*n + 1):
                if 2*s + 3*t == 9*n and 2**t <= 8**n <= 2**(2*t): c += 1
        return c
    def a_n_brute(n):                    # k,m 을 모든 약수 후보로 확인(작은 n)
        c = 0; N = 2**(9*n)
        for m in [2**t for t in range(0, 3*n + 1)] + [3, 5, 6, 7]:
            if N % (m**3) == 0:
                k2 = N // m**3; k = math.isqrt(k2)
                if k*k == k2 and m <= 8**n <= m*m: c += 1
        return c
    ok_small = all(a_n(n) == a_n_brute(n) for n in range(1, 5))
    an = [None] + [a_n(n) for n in range(1, 120)]
    closed = all(an[n] == (3*n)//4 + 1 for n in range(1, 120))
    S = lambda p: sum(an[1:p+1])
    p = max(q for q in range(1, 119) if S(q) <= 2022)
    V.check(3, "3-1 p 최댓값", sp.Integer(p), V.ans(3, "3-1"))
    V.report(3, ok_small and closed and S(72) == 2016 and an[73] == 55, f"a_n=floor(3n/4)+1 일치, S72={S(72)}, a73={an[73]}")


def p04():
    t = sp.Symbol("t"); x = sp.Symbol("x")
    # (a)
    a_ = 6*t**2 - t**3
    r1, r2 = [sp.sympify(V.ans(4, "4-1", k)) for k in ("r1", "r2")]
    # sympify 의 t 는 일반 Symbol — 같은 기호
    ok = sp.simplify(sp.expand((x - t)*(x - r1)*(x - r2)) - (x**3 - 6*x**2 + a_)) == 0
    V.report(4, ok, "4-1 (x-t)(x-r1)(x-r2) = x^3-6x^2+a")
    # (b) 수치 최대: a 를 훑으며 x1,x2 (양의 근) 직접 구해 넓이 a(x2-x1) 최대
    best = (0, None)
    N = 200000
    for i in range(1, N):
        a = 32 * i / N
        rs = sorted(r.real for r in __import__("numpy").roots([1, -6, 0, a]) if abs(r.imag) < 1e-9)
        pos = [r for r in rs if r > 0]
        if len(pos) == 2:
            area = a * (pos[1] - pos[0])
            if area > best[0]: best = (area, a)
    V.check(4, "4-2 AB=a", sp.Float(best[1], 8), V.ans(4, "4-2"), tol=2e-3)
    f = (6*T**2 - T**3) * sp.sqrt(-3*T**2 + 12*T + 36)
    crit = [c for c in sp.solve(sp.diff(sp.log(f), T), T) if -2 < float(c) < 0]
    V.check(4, "극대점 t", crit[0], 1 - sp.sqrt(7))
    V.check(4, "AB=-t^3+6t^2", (6*T**2 - T**3).subs(T, 1 - sp.sqrt(7)), V.ans(4, "4-2"))


if __name__ == "__main__": V.run(globals())
