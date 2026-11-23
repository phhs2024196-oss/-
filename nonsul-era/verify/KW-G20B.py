import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
V = Verifier("KW-G20B", "items/KW/PAST")
SY = lambda s, **kw: sp.sympify(s, locals={"t": t, "x": x, "n": n, "e": sp.E, **kw})

def p01():
    # [1](1) g=x+...+x^5, f=g^20 의 짝수 차 계수의 합
    g = sum(x**m for m in range(1, 6))
    P = sp.Poly(sp.expand(g**20), x)
    coeffs = {mon[0]: c for mon, c in P.terms()}
    V.check(1, "1-1 sum a_2n", sum(coeffs.get(2*m, 0) for m in range(1, 51)), SY(V.ans(1, "1-1")))
    # [1](2) x^2+x+1 로 나눈 나머지에 조건을 넣어 a_100 풀기
    a = sp.symbols("a0:101")
    f = sum(a[m]*x**m for m in range(1, 101))
    r = sp.Poly(sp.rem(sp.Poly(f, x), sp.Poly(x**2 + x + 1, x)), x)
    r1, r0 = r.coeff_monomial(x), r.coeff_monomial(1)
    S0 = sum(a[3*m] for m in range(1, 34)); S1 = sum(a[3*m-2] for m in range(1, 34)); S2 = sum(a[3*m-1] for m in range(1, 34))
    sol = sp.solve([r0, r1, S0 - S1, S1 - S2], [a[100], a[3], a[1], a[2]], dict=True)[0]
    V.check(1, "1-2 a100", sol[a[100]], SY(V.ans(1, "1-2")))
    # [2](2) Y∩Z
    Y = sp.Interval(1, 2)
    t_ = sp.symbols("t_", real=True)
    cond = sp.Intersection(*[sp.solveset(r_, t_, sp.S.Reals) for r_ in (-t_**2 <= t_, t_ <= t_**2, 0 < t_, t_ <= 2)])
    V.check(1, "2-2 Y=[1,2]", sp.Integer(1) if cond == Y else sp.Integer(0), 1)
    V.check(1, "2-2 Y∩Z", len([m for m in range(-10, 11) if Y.contains(m)]), 2)
    V.report(1, V.ans(1, "2-2") == "무한집합", "2-2 YAML 판정=무한집합 (R−{1,2} 는 무한)")
    # [2](1) 유한집합 X 를 빼도 실수가 계속 남는다(수치 예시)
    V.proof(1, "2-1", [X for X in ([-3, 1.5, 7], list(range(50))) if len({q for q in [k/7 for k in range(1000)]} - set(X)) <= len(X)])
    # [3](1) 작은 집합에서 일대일 g∘f=g∘f∘g 이면 g=id (전수)
    for N in (4, 5):
        perms = list(itertools.permutations(range(N)))
        bad = [(f_, g_) for f_ in perms for g_ in perms
               if all(g_[f_[i]] == g_[f_[g_[i]]] for i in range(N)) and g_ != tuple(range(N))]
        V.proof(1, f"3-1 N={N} g=id", bad)
    Xs = list(range(-5, 8))
    V.check(1, "3-1 g^-1(7)", 7 if 7 in Xs else None, SY(V.ans(1, "3-1", "ginv7")))
    V.check(1, "3-2 n(X)", len(Xs), SY(V.ans(1, "3-2", "n(X)")))

def p02():
    f = sp.sin(x) - x*sp.cos(x)
    fp = sp.simplify(sp.diff(f, x))
    V.check(2, "1-1 f'", fp - x*sp.sin(x), 0)
    cand = [k*sp.pi for k in range(-2, 3)]
    vals = [f.subs(x, c) for c in cand]
    V.check(2, "1-1 min", min(vals, key=sp.N), SY(V.ans(2, "1-1", "min")))
    V.check(2, "1-1 max", max(vals, key=sp.N), SY(V.ans(2, "1-1", "max")))
    # f' 의 부호표
    sg = [sp.sign(sp.N(fp.subs(x, c))) for c in (-3*sp.pi/2, -sp.pi/2, sp.pi/2, 3*sp.pi/2)]
    V.report(2, sg == [-1, 1, 1, -1], "1-1 부호표 −,+,+,−")
    # (2) f''<0
    f2 = sp.diff(f, x, 2)
    V.check(2, "1-2 f''", f2 - (sp.sin(x) + x*sp.cos(x)), 0)
    V.report(2, all(sp.N(f2.subs(x, sp.pi + sp.pi/2*i/200)) < 0 for i in range(1, 200)) and V.ans(2, "1-2") == "none", "1-2 (π,3π/2) 에서 f''<0, 변곡점 없음")
    # (3) g=f∘f, g(π), g(2π), g' = 1 인 점
    gg = f.subs(x, f)
    V.check(2, "1-3 g(2π)", sp.simplify(gg.subs(x, 2*sp.pi)), 2*sp.pi)
    V.check(2, "1-3 g(π)", sp.simplify(gg.subs(x, sp.pi)), sp.pi)
    h = sp.diff(gg, x) - 1
    xs = [sp.pi + sp.pi*i/400 for i in range(1, 400)]
    sv = [sp.N(h.subs(x, v)) for v in xs]
    V.report(2, any(a_*b_ <= 0 for a_, b_ in zip(sv, sv[1:])), "1-3 g'(x)-1 이 (π,2π) 에서 부호 변화(실제 해 존재)")
    # [2]
    S = lambda m: sp.integrate(sp.log(x)**m, (x, 1, sp.E))
    V.check(2, "2-1 S1", S(1), SY(V.ans(2, "2-1")))
    for m in (1, 2, 3, 4):
        V.check(2, f"2-2 n={m}", sp.simplify(S(m+1) + (m+1)*S(m)), SY(V.ans(2, "2-2")))
    Vv = sp.integrate(sp.pi/2*(sp.log(x)/2)**2, (x, 1, sp.E))
    V.check(2, "2-3 V", sp.simplify(Vv), SY(V.ans(2, "2-3")))
    # [3]
    F = SY(V.ans(2, "3", "f")); G = SY(V.ans(2, "3", "g"))
    V.check(2, "3 (가)", sp.simplify(sp.exp(x)*(F + sp.diff(F, x)) - G), 0)
    tt = sp.symbols("tt", real=True)
    V.check(2, "3 (나)", sp.simplify(sp.integrate(G.subs(x, tt), (tt, 0, x)) - (sp.exp(x**2) - F.subs(x, 0))), 0)

if __name__ == "__main__": V.run(globals())
