import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DDW-G26", "items/DDW/PAST")
E_ = sp.E


def p01():
    # 1-1: f(x)=1+c(x-e)+(x-e)^2 꼴(미분가능, f(e)=1)에 대해 극한=5 가 되는 c 를 풀어 f'(e) 와 비교
    c = sp.symbols("c")
    fx = 1 + c*(x - E_) + (x - E_)**2
    lim = sp.limit((fx - 1)/(sp.log(x) - 1), x, E_)
    sol = sp.solve(sp.Eq(lim, 5), c)[0]
    V.check(1, "1-1 f'(e)", sol, V.ans(1, "1-1", "fpe"))
    V.check(1, "1-1 f(e)", fx.subs({x: E_}), V.ans(1, "1-1", "fe"))
    # 확인: YAML 의 f'(e) 로 만든 함수가 극한 5 를 준다
    fy = 1 + sp.sympify(V.ans(1, "1-1", "fpe"))*(x - E_)
    V.check(1, "1-1 극한=5", sp.limit((fy - 1)/(sp.log(x) - 1), x, E_), 5)
    # 1-2: h'/h 의 x->0 극한을 직접 계산
    h = sp.prod([1 + sp.exp(j*x) for j in range(1, 11)])
    V.check(1, "1-2 lim h'/h", sp.limit(sp.diff(h, x)/h, x, 0), V.ans(1, "1-2"))


def p02():
    # 원에 내접하는 사각형 AB=2, BC=8-x, CD=4, DA=x. Brahmagupta 로 S^2, 외접반지름은 Parameshvara 공식으로 독립 계산
    s = sp.Rational(2 + 4 + 8, 2)
    S2 = (s - 2)*(s - (8 - x))*(s - 4)*(s - x)
    S2 = sp.expand(S2)
    xs = sp.solve(sp.diff(S2, x), x)
    assert xs == [4] and 1 < xs[0] < 7
    S2m = S2.subs(x, 4)
    a_, b_, c_, d_ = 2, 8 - x, 4, x
    R2 = ((a_*b_ + c_*d_)*(a_*c_ + b_*d_)*(a_*d_ + b_*c_))/(16*S2)
    R2m = sp.simplify(R2.subs(x, 4))
    V.check(2, "2-1 S^2max", S2m, 135)
    V.check(2, "2-1 R^2", R2m, sp.Rational(32, 5))
    V.check(2, "2-1 (SR)^2", S2m*R2m, V.ans(2, "2-1"))
    # 공식 해설 경로: cosθ=(4x-19)/(16-x), BD^2, sin^2θ 로 R^2 재계산
    th = (4*x - 19)/(16 - x)
    BD2 = 4 + x**2 - 4*x*th
    V.check(2, "2-1 S^2(x)=-15(x-1)(x-7)", sp.simplify((16 - x)**2*(1 - th**2) - (-15*(x - 1)*(x - 7))), 0)
    V.check(2, "2-1 R^2 via sine law", sp.simplify((BD2/(4*(1 - th**2))).subs(x, 4)), sp.Rational(32, 5))


def p03():
    a, b, cc = sp.symbols("a b cc")
    fg = x**4 + a*x**2 + b
    # g(x)=x^2+cc*x+1 가 함수방정식을 만족함을 확인(미분가능 해는 이 꼴), g(0)=1
    gx = lambda z: z**2 + cc*z + 1
    V.check(3, "3 함수방정식", sp.expand(gx(x + y) - (gx(x) + gx(y) + 2*x*y - 1)), 0)
    # 극한 조건: (f g - 3)/x -> 0  <=> f(0)g(0)=3 이고 (fg)'(0)=0
    b_sol = sp.solve(sp.Eq(fg.subs(x, 0)*gx(0), 3), b)[0]
    fx = fg.subs(b, b_sol)
    # 극솟값 -13
    a_sols = sp.solve(sp.Eq(fx.subs(x, sp.sqrt(-a/2)), -13), a)
    a_neg = [s_ for s_ in a_sols if s_ < 0]
    assert len(a_neg) == 1
    f_final = sp.expand(fx.subs(a, a_neg[0]))
    # 극솟값 실제 확인
    mins = [f_final.subs(x, r) for r in sp.solve(sp.diff(f_final, x), x) if sp.diff(f_final, x, 2).subs(x, r) > 0]
    V.check(3, "3-1 극솟값=-13", set(mins), {sp.Integer(-13)})
    V.check(3, "3-1 f", f_final, sp.sympify(V.ans(3, "3-1"), locals={"x": x}))
    # 3-2: 극한 = (fg)'(0) = 0 이 되도록 cc 결정
    d0 = sp.diff(f_final*gx(x), x).subs(x, 0)
    csol = sp.solve(sp.Eq(d0, 0), cc)[0]
    V.check(3, "3-2 g'(0)", sp.diff(gx(x), x).subs({x: 0, cc: csol}), V.ans(3, "3-2"))
    V.check(3, "3-2 극한 확인", sp.limit((f_final*gx(x).subs(cc, csol) - 3)/x, x, 0), 0)

if __name__ == "__main__": V.run(globals())
