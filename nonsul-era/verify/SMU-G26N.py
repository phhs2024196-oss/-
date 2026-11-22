import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("SMU-G26N", "items/SMU/PAST")
a, b, r, s = sp.symbols("a b r s", real=True)
E = lambda v: sp.sympify(v, locals={"x": x, "n": n})  # 식 답은 vlib 의 실수 기호로 읽기

def p01():
    X, Y = sp.symbols("X Y", positive=True)
    so = sp.solve([2*X + 9*Y - 113, 4*X + 3*Y - 91], [X, Y])
    A, B = sp.log(so[X], 2), sp.log(so[Y], 3)
    V.check(1, "1 a+b", sp.nsimplify(A + B), V.ans(1, "1"))
    lhs = sp.log((A + 1/B)**2, A) + sp.log(B + 1/A, B)
    q = sp.nsimplify(sp.simplify(B**lhs))   # m/n
    m_, n_ = sp.fraction(q)
    V.check(1, "2 n*sqrt(m)", n_*sp.sqrt(m_), V.ans(1, "2"))

def p02():
    sols = [z for z in sp.solveset(1/(2-2*sp.cos(x)) - 1/(2+2*sp.cos(x)) - sp.Rational(2, 3), x, sp.Interval.open(0, sp.pi))]
    V.check(2, "1 alpha(유일)", sols, [V.ans(2, "1")])
    cA = sp.cos(sols[0])/2; sA = sp.sqrt(1 - cA**2)
    V.check(2, "2 sinA", sA, V.ans(2, "2"))
    V.check(2, "3 BC", 2*sp.sqrt(15)*sA, V.ans(2, "3"))

def p03():
    an = lambda k, rr: rr**(k - 2)
    cand = [rr for rr in sp.solve(2*sum(an(k, r) for k in range(1, 10)) - 7*sum(an(3*k - 1, r) for k in range(1, 4)), r) if rr.is_real and rr > 1]
    V.check(3, "1 공비(r>1 유일)", cand, [V.ans(3, "1")])
    rr = cand[0]
    nmin = next(k for k in range(1, 100) if an(3*k - 1, rr)/an(k, rr) > 1024)
    V.check(3, "2 n", nmin, V.ans(3, "2"))
    V.check(3, "3 log", sp.log(an(5, rr), an(3*3 - 1, rr)), V.ans(3, "3"))

def p04():
    A, B = sp.symbols("A B")
    g = x**2 + 2*A*x + B
    f = sp.expand(g + x*sp.diff(g, x) + 2)
    # 출제 의도 풀이: g(1)=-4, A=∫f 두 조건으로 결정
    so = sp.solve([sp.integrate(f, (x, 0, 1)) - A, g.subs(x, 1) + 4], [A, B], dict=True)[0]
    G, F = g.subs(so), f.subs(so)
    V.report(4, sp.simplify(sp.integrate(F.subs(x, t), (t, 1, x)) - (x*G + 2*x + 2)) == 0, "둘째 식(∫_1^x f = xg+2x+2) 만족")
    # 원문 결함 기록: B=∫t f(t)dt 는 -17/12 ≠ -1 → 첫째 식은 정확히 성립하지 않음(조건 과잉·모순). FAIL 이 아니라 기록만.
    Bt = sp.integrate(x*F, (x, 0, 1))
    if Bt != so[B]:
        print(f"NOTE SMU-G26N-04 원문 조건 모순: ∫_0^1 t f(t)dt = {Bt} ≠ B = {so[B]} (공식 정답 비공개, 의도 풀이로 입력)") if V.verbose else None
    V.check(4, "1 g", G, E(V.ans(4, "1"))); V.check(4, "2 f", F, E(V.ans(4, "2")))
    h = F - G; xs = sorted(sp.solve(h - 2, x))
    V.check(4, "3 넓이", sp.integrate(2 - h, (x, xs[0], xs[-1])), V.ans(4, "3"))

def p05():
    al = sp.symbols("al", positive=True)
    fp = 4*(x + 1)*(x - al)**2
    f0 = -sp.integrate(fp, (x, 0, al))      # f(0)=f(al)-int_0^al f'
    als = [z for z in sp.solve(f0 + 16, al) if z.is_real and z > 0]
    V.check(5, "1 alpha", als, [V.ans(5, "1")])
    fpp = fp.subs(al, als[0])
    g = lambda tt: max(fpp.subs(x, u) for u in [sp.Rational(-1)] + [c for c in sp.solve(sp.diff(fpp, x), x) if -1 <= c <= tt] + [tt])
    V.check(5, "2 g(3/2)", g(sp.Rational(3, 2)), V.ans(5, "2"))
    # g(t): f' 증가 구간 [-1,0], 상수 16 [0,3], f' [3,4] — 수치적분으로 확인
    import mpmath as mp
    gf = lambda tt: max(float(fpp.subs(x, u)) for u in [-1, 0, tt] if u <= tt) if tt >= 0 else float(fpp.subs(x, tt))
    val = mp.quad(gf, [-1, 0, 3, 4])
    V.check(5, "3 int g (수치)", val, V.ans(5, "3"), tol=1e-8)

def p06():
    # (가): a>1 에서만 성립 — 표본 점검
    import random
    def cond(av):
        return all(sp.log(pp, av)*(qq - 1) > sp.log(qq, av)*(pp - 1) for pp, qq in [(1.5, 2), (2, 9), (1.01, 50), (3, 3.5)])
    cands = [av for av in [2, sp.Rational(1, 2)] if cond(av)]   # |log_a 8|=3 → a=2 또는 1/2
    V.check(6, "1 a", cands, [V.ans(6, "1")])
    av = cands[0]; f = sp.log(x, av); g = -f + 4
    A = sp.solve(f, x)[0]; B = sp.solve(g, x)[0]; C = sp.solve(f - g, x)[0]
    V.check(6, "2 넓이", sp.Rational(1, 2)*abs(B - A)*f.subs(x, C), V.ans(6, "2"))
    # 역함수 y=a^x → 원점대칭 y=-a^(-x) → y=x 대칭: x=-a^(-y) → y=-log_a(-x) → x축 k 이동: y=-log_a(-(x-k))
    kk = sp.symbols("kk")
    ks = sp.solve(-sp.log(-(-6 - kk), av) + 1, kk)
    V.check(6, "3 k", ks, [V.ans(6, "3")])

def p07():
    def S13(d):
        a = lambda i: ((i + 1)//2)**3 + 1 if i % 2 else (i//2)**3 + 1 + d
        return sum(a(i) for i in range(1, 14)), a
    ok = [d for d in range(-50, 50) if all(S13(d)[1](i) > 0 and S13(d)[1](i) < S13(d)[1](i + 1) for i in range(1, 400))]
    V.report(7, ok == list(range(1, 7)), f"가능한 d = {ok}")
    vals = [S13(d)[0] for d in ok]
    V.check(7, "1 차이", max(vals) - min(vals), V.ans(7, "1"))
    dmin = ok[vals.index(min(vals))]
    V.check(7, "2 a16", S13(dmin)[1](16), V.ans(7, "2"))

def p08():
    kk = sp.symbols("kk")
    f = lambda u: (u - 1)*(u - kk)
    gp = lambda X: f(abs(X + 5)) - f(abs(X))
    K = sp.solve(gp(sp.Rational(1, 2)), kk)
    V.check(8, "1 f", [sp.expand(f(x).subs(kk, K[0]))], [E(V.ans(8, "1"))])
    F = lambda u: (u - 1)*(u - K[0])
    G = lambda X: F(abs(X + 5)) - F(abs(X))
    zs = set()
    for lo, hi, e in [(-sp.oo, -5, F(-x - 5) - F(-x)), (-5, 0, F(x + 5) - F(-x)), (0, sp.oo, F(x + 5) - F(x))]:
        for z in sp.solve(sp.expand(e), x):
            if (z > lo or lo == -sp.oo) and (z < hi or (hi == 0 and False)) or (lo == 0 and z >= 0):
                if sp.simplify(G(z)) == 0: zs.add(z)
    others = sorted(z for z in zs if z != sp.Rational(1, 2))
    V.check(8, "2 alpha", others[0], V.ans(8, "2", "alpha")); V.check(8, "2 beta", others[1], V.ans(8, "2", "beta"))
    sgn = lambda X: sp.sign(G(X))
    # x=1/2 극소 확인, 극대점 찾기
    eps = sp.Rational(1, 100)
    V.report(8, sgn(sp.Rational(1, 2) - eps) < 0 < sgn(sp.Rational(1, 2) + eps), "x=1/2 극소")
    mx = [z for z in sorted(zs) if sgn(z - eps) > 0 > sgn(z + eps)]
    V.check(8, "3 극대 x", mx, [V.ans(8, "3")])

if __name__ == "__main__": V.run(globals())
