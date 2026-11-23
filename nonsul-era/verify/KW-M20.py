import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
V = Verifier("KW-M20", "items/KW/PAST")
SY = lambda s, **kw: sp.sympify(s, locals={"t": t, "x": x, "n": n, "e": sp.E, **kw})

def p01():
    # [1] 모든 x 에서 성립하는 k: 격자 k=-20..20(1/8 간격) 에서 x=0, ±10^9 로 검사
    def ok(kv):
        F = lambda xv: kv*xv - abs(kv)*xv - kv**2 - kv + 56
        return all(F(xv) >= 0 for xv in (0, 10**9, -10**9, 1, -1))
    good = [sp.Rational(i, 8) for i in range(-160, 161) if ok(sp.Rational(i, 8))]
    V.check(1, "1 min", min(good), SY(V.ans(1, "1", "min")))
    V.check(1, "1 max", max(good), SY(V.ans(1, "1", "max")))
    # [2](2) B=(1,∞)
    t_ = sp.symbols("t_", real=True)
    B = sp.Intersection(sp.solveset(t_ > 0, t_, sp.S.Reals), sp.solveset(t_ < t_**2, t_, sp.S.Reals))
    V.check(1, "2-2 B=(1,oo)", 1 if B == sp.Interval.open(1, sp.oo) else 0, 1)
    V.report(1, V.ans(1, "2-2") == "무한집합" and B.is_subset(sp.S.Reals) and B.measure == sp.oo, "2-2 판정=무한집합")
    # [2](1) +10 에 닫힌 집합은 유한일 수 없다(수치): 원소 a 에서 시작해 a+10m 이 모두 서로 다름
    V.proof(1, "2-1", [m for m in range(1, 50) if sp.Rational(1, 3) + 10*m in {sp.Rational(1, 3) + 10*q for q in range(m)}])
    # [3] 일대일 대응(순열) 전수: 3-1 역함수 단사, 3-2 f∘g=f∘h ⇒ g=h, 3-3 명제
    for N in (4, 5):
        perms = list(itertools.permutations(range(N)))
        inv = lambda p: tuple(sorted(range(N), key=lambda i: p[i]))
        V.proof(1, f"3-1 N={N}", [p for p in perms if len(set(inv(p))) != N])
        V.proof(1, f"3-2 N={N}", [(f_, g_, h_) for f_ in perms for g_ in perms[:12] for h_ in perms[:12]
                                    if all(f_[g_[i]] == f_[h_[i]] for i in range(N)) and g_ != h_])
        bad = []
        for f_ in perms:
            fi = inv(f_)
            prem = any(f_[f_[a]] != a for a in range(N))
            concl = any(fi[fi[c]] != c for c in range(N))
            if prem and not concl: bad.append(f_)
        V.proof(1, f"3-3 N={N} 명제", bad)
    V.check(1, "3-2 h(5)", 5**3, SY(V.ans(1, "3-2")))
    V.report(1, V.ans(1, "3-3").endswith("참"), "3-3 YAML 판정=참")

def p02():
    h = lambda tv: tv + 2*abs(tv - 2)
    def F(xv):  # 정확한 적분(조각 선형이므로 끝점·꺾임점 분할)
        pts = sorted({xv - 2, xv, sp.Integer(2)} & {q for q in [xv - 2, 2, xv] if xv - 2 <= q <= xv})
        return sum(sp.integrate(sp.Piecewise((-t + 4, t <= 2), (3*t - 4, True)), (t, a, b)) for a, b in zip(pts, pts[1:]))
    # (1)
    for xv in (-3, 0, 1, sp.Rational(3, 2), 2):
        V.check(2, f"1-1 f({xv})", F(sp.Integer(xv) if not isinstance(xv, sp.Rational) else xv), -2*xv + 10)
    V.check(2, "1-1 극한", sp.limit((-2*(2*x) + 10 - (-2*sp.sin(x) + 10))/x, x, 0), SY(V.ans(2, "1-1")))
    # (2) 도함수: 수치 미분
    ans12 = SY(V.ans(2, "1-2"), Abs=sp.Abs)
    d = sp.Rational(1, 10**6)
    for xv in (-1, 1, sp.Rational(5, 2), 3, 5, 7):
        xv = sp.Integer(xv) if not isinstance(xv, sp.Rational) else xv
        nd = (F(xv + d) - F(xv - d))/(2*d)
        V.check(2, f"1-2 f'({xv})", sp.N(nd), sp.N(ans12.subs(x, xv)), tol=1e-5)
    # (3) 삼각형 PAB
    quad = 2*x**2 - 10*x + 18
    for xv in (2, sp.Rational(5, 2), 3, 4):
        V.check(2, f"1-3 f({xv}) 이차식", F(sp.Integer(xv) if not isinstance(xv, sp.Rational) else xv), quad.subs(x, xv))
    A = sp.Matrix([3, 0]); B = sp.Matrix([4, 2]); P = sp.Matrix([x, quad])
    cross = (B - A)[0]*(P - A)[1] - (B - A)[1]*(P - A)[0]
    V.report(2, sp.simplify(cross - (2*x**2 - 12*x + 24)) == 0 and all(cross.subs(x, v) > 0 for v in (2, 3, 4)), "1-3 외적 부호 양")
    area = cross/2
    mn, mx = extrema(area, x, 2, 4)
    V.check(2, "1-3 min", mn, SY(V.ans(2, "1-3", "min")))
    xs = sp.solve(sp.diff(area, x), x)[0]
    V.check(2, "1-3 P", f"({xs},{quad.subs(x, xs)})", V.ans(2, "1-3", "P")) if False else V.report(2, f"({xs},{quad.subs(x, xs)})" == V.ans(2, "1-3", "P"), f"1-3 P=({xs},{quad.subs(x, xs)})")
    # (4) 전 구간 최솟값: 조각별 식을 정확 적분과 대조 후 격자 최솟값
    def f_piece(xv):
        return -2*xv + 10 if xv <= 2 else (quad.subs(x, xv) if xv <= 4 else 6*xv - 14)
    for xv in (-4, 0, sp.Rational(7, 2), 5, 9):
        xv = sp.Integer(xv) if not isinstance(xv, sp.Rational) else xv
        V.check(2, f"1-4 조각식 f({xv})", F(xv), f_piece(xv))
    grid = [sp.Rational(i, 8) for i in range(-80, 161)]
    vals = [f_piece(g) for g in grid]
    V.check(2, "1-4 min", min(vals), SY(V.ans(2, "1-4", "min")))
    V.check(2, "1-4 argmin", grid[vals.index(min(vals))], sp.Rational(5, 2))
    # [2](1)
    a_, b_, c_ = sp.symbols("a b c", real=True)
    g = x**3 + a_*x**2 + b_*x + c_
    rem = sp.Poly(sp.rem(sp.Poly(g, x), sp.Poly(sp.diff(g, x), x)), x)
    sols = sp.solve([*rem.coeffs(), c_ - 27], [a_, b_, c_], dict=True)
    sols = [s_ for s_ in sols if all(v.is_real for v in s_.values())]
    G = SY(V.ans(2, "2-1"))
    V.check(2, "2-1 해의 개수", len(sols), 1)
    V.check(2, "2-1 g", sp.expand(g.subs(sols[0])), sp.expand(G))
    # [2](2)
    gp = sp.diff(G, x)
    pts = sp.solve(sp.Eq(G, gp), x)
    V.check(2, "2-2 교점", sorted(pts), [-3, 0])
    V.check(2, "2-2 넓이", sp.integrate(sp.Abs(sp.expand(gp - G)), (x, -3, 0)) if False else sp.integrate(gp - G, (x, -3, 0)), SY(V.ans(2, "2-2")))
    V.report(2, all(sp.N((gp - G).subs(x, v)) >= 0 for v in (-2.9, -2, -1, -0.1)), "2-2 [-3,0] 에서 g'>=g")

if __name__ == "__main__": V.run(globals())
