import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
E = lambda v: sp.sympify(v, locals={"x": x, "n": n})  # 식 답은 vlib 의 실수 기호로 읽기

V = Verifier("SMU-G26H", "items/SMU/PAST")
p, q, r = sp.symbols("p q r", real=True)

def p09():
    sols = []
    for pp in sp.solve(2*abs(p) - 20, p):          # 최댓값-최솟값 = 2|p|
        qq = sp.solve(pp*sp.sin(sp.pi/6) + q - 3, q)[0]
        for rr in sp.solve(sp.pi/abs(r) - 2*sp.pi, r):  # |cos(rx)| 주기 pi/|r| = 2pi
            sols.append((pp, qq, rr))
    ps = sorted(set(s[0] for s in sols)); qs = sorted(set(s[1] for s in sols))
    V.check(9, "1 p곱", sp.prod(ps), V.ans(9, "1"))
    V.check(9, "2 q합", sum(qs), V.ans(9, "2"))
    V.check(9, "3 pr 최소", min(s[0]*s[2] for s in sols), V.ans(9, "3"))
    # 주기 수치 확인
    g = lambda X: abs(sp.cos(sp.Rational(1, 2)*X))
    V.report(9, all(sp.N(g(u) - g(u + 2*sp.pi)) == 0 for u in [0.3, 1.1]) and sp.N(g(0.3) - g(0.3 + sp.pi)) != 0, "|cos(x/2)| 기본주기 2pi")

def p10():
    a3, a2, a1 = sp.symbols("a3 a2 a1")
    f = x**4 + a3*x**3 + a2*x**2 + a1*x        # f(0)=0
    F = E(V.ans(10, "2"))
    V.check(10, "2 f", F, E(V.ans(10, "2")))
    V.check(10, "f(0)=0, lim f/x=6", sp.limit(F/x, x, 0), 6)
    rts = [z for z in sp.Poly(F, x).nroots() if abs(sp.im(z)) < 1e-12]
    c = sp.limit(x*(x + 2)/F, x, 0)
    V.report(10, all(abs(sp.N(sp.limit(x*(x + 2)/F, x, sp.nsimplify(sp.re(z))) - c)) < 1e-12 for z in rts), f"모든 실근 {rts} 에서 극한 = c")
    V.check(10, "1 c", c, V.ans(10, "1"))
    # 유일성: f=x(x+2)(x^2+bx+h0) 꼴에서 f'(0)=6, h(-2)=3 → 계수 유일
    b_, h0 = sp.symbols("b_ h0")
    cand = sp.expand(x*(x + 2)*(x**2 + b_*x + h0))
    so = sp.solve([sp.diff(cand, x).subs(x, 0) - 6, (x**2 + b_*x + h0).subs(x, -2) - 3], [b_, h0], dict=True)
    V.check(10, "2 f 유일해", [sp.expand(cand.subs(s)) for s in so], [F])

if __name__ == "__main__": V.run(globals())
