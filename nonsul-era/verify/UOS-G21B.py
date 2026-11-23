import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("UOS-G21B", "items/UOS/PAST")


def p01():
    a = sp.symbols("a", positive=True)
    m, al, be = sp.symbols("m alpha beta", real=True)
    f = x**4 + 2*a*x**3 - 3*a**2*x**2 + 4*a**4 - 4*a**3 + 1
    # f - (m(x-1)+1) = (x-al)^2 (x-be)^2 , 계수 비교
    diff = sp.expand(f - (m*(x - 1) + 1) - (x - al)**2*(x - be)**2)
    eqs = sp.Poly(diff, x).all_coeffs()
    sol = sp.solve(eqs, [m, al, be], dict=True)
    sol = [s for s in sol if sp.simplify(s[al] - s[be]) != 0 and sp.simplify(s[al] - s[be]) .is_negative]
    ok = len(sol) == 1 and same(sol[0][m], 4*a**3) and same(sol[0][al], -2*a) and same(sol[0][be], a)
    V.report(1, ok, f"접점 조건 연립 해 {sol}")
    s = sol[0]
    area = sp.integrate(sp.expand(f - (s[m]*(x - 1) + 1)), (x, s[al], s[be]))
    V.check(1, "1-1 넓이", area, sp.sympify(V.ans(1, "1-1"), locals={"a": a}))


def p02():
    expr = S(V.ans(2, "2-1"))
    n_ = sp.symbols("n")
    for N in range(9, 13):
        cnt = 0
        for f in itertools.product((0, 1, 2), repeat=N):
            if f.count(0) != 3 or f.count(1) != 3:
                continue
            s, ok = 0, True
            for i, v in enumerate(f, 1):
                s += v
                if s < i:
                    ok = False
                    break
            cnt += ok
        V.check(2, f"2-1 n={N} 전수 {cnt}", cnt, expr.subs(n_, N))
    # 해설의 중간 결과: 1이 마지막 세 칸일 때의 개수
    N = 11
    cnt = 0
    for f in itertools.product((0, 1, 2), repeat=N - 3):
        g = f + (1, 1, 1)
        if g.count(0) != 3 or g.count(1) != 3:
            continue
        s, ok = 0, True
        for i, v in enumerate(g, 1):
            s += v
            if s < i:
                ok = False
                break
        cnt += ok
    V.check(2, "중간식 (n-3)(n-4)(n-8)/6", cnt, sp.Rational((N - 3)*(N - 4)*(N - 8), 6))


def p03():
    # (b) 두 곡선 사이 최단거리: 수치 + 해설 조건
    import math
    best = min(math.hypot(p - q, p*p + (q - 6)**2) for p in [i/100 for i in range(-100, 400)] for q in [j/100 for j in range(200, 800, 1)])
    V.report(3, abs(best - float(S(V.ans(3, "3-2")))) < 2e-2, f"3-2 격자 최소 {best:.4f} / 정답 {float(S(V.ans(3, '3-2'))):.4f}")
    a, b = sp.symbols("a b", real=True)
    sol = sp.solve([2*a + 2*(b - 6), (a**2 + (b - 6)**2)*2*a + (a - b)], [a, b], dict=True)  # ①: 기울기 같음, ②: 접선에 수직
    sols = [s for s in sol if s[a].is_real and s[b].is_real and s[a] != s[b]]
    dist = sp.sqrt((sols[0][a] - sols[0][b])**2 + (sols[0][a]**2 + (sols[0][b] - 6)**2)**2)
    V.check(3, "3-2 (a,b)=(1,5) 후 AB", dist, V.ans(3, "3-2"))
    V.check(3, "3-2 인수분해 2a^3+a-3", sp.factor(2*a**3 + a - 3), (a - 1)*(2*a**2 + 2*a + 3))
    # (a) 수치확인: f=x^2, Q=(0,3)에서 가장 가까운 점 P 에서 접선 ⟂ PQ
    f = x**2
    Q = (0, 3)
    g = (x - Q[0])**2 + (f - Q[1])**2
    crit = [r for r in sp.solve(sp.diff(g, x), x) if r.is_real]
    p0 = min(crit, key=lambda r: g.subs(x, r))
    dot = (1)*(p0 - Q[0]) + sp.diff(f, x).subs(x, p0)*(f.subs(x, p0) - Q[1])
    V.check(3, "3-1 수치확인 (접선 방향·PQ 내적=0)", dot, 0)


def p04():
    def icbrt_ceil(v):  # ceil of cube root of positive int v
        lo, hi = 0, 1 << (v.bit_length()//3 + 2)
        while lo < hi:
            mid = (lo + hi)//2
            if mid**3 >= v:
                hi = mid
            else:
                lo = mid + 1
        return lo
    def total(nexp):
        n_ = 2**nexp
        tot = 0
        for m in range(1, nexp + 3):
            # k^3 >= 2^(9m-12) 이고 k^3 < 2^(9m)  (8^m k^3 가 [64^(2m-2), 64^(2m)) 안), 1<=k<=n
            lo = max(1, icbrt_ceil(2**(9*m - 12)) if 9*m - 12 >= 0 else 1)
            hi = icbrt_ceil(2**(9*m)) - 1
            hi = min(hi, n_)
            if hi >= lo:
                tot += hi - lo + 1
        return tot
    # 작은 n 에서 정의대로(교점 존재 판정) 직접 세어 같은 식과 비교
    for nexp in (6, 9):
        n_ = 2**nexp
        direct = 0
        for k in range(1, n_ + 1):
            for m in range(1, nexp + 3):
                if sp.Integer(64)**(2*m - 2) <= sp.Integer(8)**m*k**3 < sp.Integer(64)**(2*m):
                    direct += 1
        V.check(4, f"n=2^{nexp} 직접 개수", total(nexp), direct)
    V.check(4, "4-1 n=2^300", total(300), V.ans(4, "4-1"))


if __name__ == "__main__":
    V.run(globals())
