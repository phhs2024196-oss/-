import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import math
import mpmath as mp
V = Verifier("KW-G16B", "items/KW/PAST")
M = sp.Matrix
SY = lambda s, **kw: sp.sympify(s, locals={"t": t, "x": x, "n": n, "Matrix": M, **kw})
A_ = M([[sp.cos(sp.pi/4), -sp.sin(sp.pi/4)], [sp.sin(sp.pi/4), sp.cos(sp.pi/4)]])
B_ = M([[1/sp.sqrt(2), 0], [0, 1/sp.sqrt(2)]])
H = sp.simplify(B_*A_)

def p01():
    Hy = SY(V.ans(1, "1", "H"))
    V.report(1, sp.simplify(H - Hy) == sp.zeros(2), "1 H=BA 가 YAML 과 일치")
    Hn = SY(V.ans(1, "1", "Hn"))
    V.report(1, all(sp.simplify((H**k - Hn.subs(n, k))) == sp.zeros(2) for k in range(1, 13)), "1 H^n 공식 (n=1..12) 행렬 거듭제곱과 일치")
    # [1] 별해2: H^1..H^8
    claims = {1: sp.Rational(1, 2)*M([[1, -1], [1, 1]]), 2: sp.Rational(1, 2)*M([[0, -1], [1, 0]]), 3: sp.Rational(1, 4)*M([[-1, -1], [1, -1]]),
              4: sp.Rational(1, 4)*M([[-1, 0], [0, -1]]), 5: sp.Rational(1, 8)*M([[-1, 1], [-1, -1]]), 6: sp.Rational(1, 8)*M([[0, 1], [-1, 0]]),
              7: sp.Rational(1, 16)*M([[1, 1], [-1, 1]]), 8: sp.Rational(1, 16)*M([[1, 0], [0, 1]])}
    V.report(1, all(sp.simplify(H**k - c) == sp.zeros(2) for k, c in claims.items()), "1 별해2 H^1~H^8 나열")
    # [2] R, S, 원, 길이
    Q = M([1, 1]); R = H*Q; S = H.inv()*Q
    V.report(1, list(R) == [0, 1] and list(S) == [2, 0], f"2 R={list(R)}, S={list(S)}")
    X, Y = sp.symbols("X Y", real=True)
    dot = sp.expand((R[0] - X)*(S[0] - X) + (R[1] - Y)*(S[1] - Y))
    circ = sp.expand((X - 1)**2 + (Y - sp.Rational(1, 2))**2 - sp.Rational(5, 4))
    V.check(1, "2 PR·PS=0 ⇔ 원", sp.simplify(dot - circ), 0)
    V.check(1, "2 길이", 2*sp.pi*sp.sqrt(sp.Rational(5, 4)), SY(V.ans(1, "2")))
    # [3] S_n 합
    P = [M([1, 0])]
    for k in range(1, 61): P.append(H*P[-1])
    Sn = [sp.Abs(P[k-1][0]*P[k][1] - P[k-1][1]*P[k][0])/2 for k in range(1, 61)]
    V.report(1, all(Sn[k] == sp.Rational(1, 2)*Sn[k-1] for k in range(1, 20)) and Sn[0] == sp.Rational(1, 4), "3 S_1=1/4, 공비 1/2 (정확값 20항)")
    V.check(1, "3 부분합 → 합", sp.N(sum(Sn), 30), sp.N(SY(V.ans(1, "3")), 30), tol=1e-15)
    V.report(1, list(P[1]) == [sp.Rational(1, 2), sp.Rational(1, 2)] and list(P[2]) == [0, sp.Rational(1, 2)] and list(P[3]) == [-sp.Rational(1, 4), sp.Rational(1, 4)] and list(P[4]) == [-sp.Rational(1, 4), 0],
             "3 별해1 점들(원문 표기 P3(-1/4,-1/4) 는 계산상 (-1/4,1/4))")
    # [4]
    a = sp.symbols("a", positive=True)
    sol = [r for r in sp.solve(a**4 - 4*a**2 - 1, a)]
    a2 = [sp.simplify(r**2) for r in sol if r.is_real and r > 0]
    V.check(1, "4 a^2", a2[0], SY(V.ans(1, "4")))
    av = sp.sqrt(a2[0]); Dp = H*M([av, 1/av])
    V.check(1, "4 h(C)=(b,1/b)", sp.simplify(Dp[0]*Dp[1]), 1)
    # [5]
    L0 = [M([1, 0]), M([0, 1])]
    ms = []
    for k in range(1, 13):
        Hk = H**k; u = Hk*(L0[1] - L0[0])
        if sp.simplify(u[0] + u[1]) == 0: ms.append(k)   # u ∥ (1,-1)
    V.report(1, ms[:3] == [4, 8, 12], f"5 평행이 되는 n = {ms}")
    P4 = H**4*L0[0]; P4b = H**4*L0[1]
    nvec = sp.Matrix([1, 1]); dist = sp.Abs(nvec.dot(P4) - 1)/sp.sqrt(2)
    V.check(1, "5 l_0 와 l_4 거리", sp.simplify(dist), SY(V.ans(1, "5")))
    V.report(1, sp.simplify(H**4 + sp.eye(2)/4) == sp.zeros(2), "5 H^4=-E/4")
    V.report(1, sp.simplify(4*P4[0] + 4*P4[1] + 1) == 0 and sp.simplify(4*P4b[0] + 4*P4b[1] + 1) == 0, "5 l_4: 4x+4y=-1")

def p02():
    mp.mp.dps = 30
    # [1]
    nn = 400000
    s1 = float(sum(math.sqrt(nn*k) for k in range(2*nn + 1, 3*nn + 1)))/nn**2
    V.check(2, "1 급수 수치", s1, sp.N(SY(V.ans(2, "1")), 20), tol=1e-5)
    V.check(2, "1 적분 두 꼴", sp.integrate(sp.sqrt(2 + x), (x, 0, 1)), sp.integrate(sp.sqrt(x), (x, 2, 3)))
    V.check(2, "1 정적분 값", sp.integrate(sp.sqrt(x), (x, 2, 3)), SY(V.ans(2, "1")))
    # [2]
    h = lambda xx: mp.sqrt(1 + 4/xx**4)
    Lf = lambda tt: mp.quad(h, [tt, tt + 1])
    for tv in (1, 2.5, 4):
        V.check(2, f"2 L'({tv})", mp.diff(Lf, tv), h(tv + 1) - h(tv), tol=1e-12)
    nv = 7
    V.check(2, "2 합(n=7)", sum(mp.diff(Lf, k) for k in range(1, nv + 1)), sp.N(SY(V.ans(2, "2")).subs(n, nv), 25), tol=1e-10)
    # [3] 수치확인: f(t+h)-f(t)=h f'(c) 인 c in (t,t+h)
    bad = []
    for f_ in (sp.exp(x), x**3, sp.sin(x) + x**2):
        for tv in (0.3, 1.0, 2.0):
            for hv in (0.5, 1.0, 2.0):
                fv = sp.lambdify(x, f_, "mpmath"); dfv = sp.lambdify(x, sp.diff(f_, x), "mpmath")
                tt_ = sp.symbols("tt")
                der = mp.diff(lambda u: mp.quad(fv, [u, u + hv]), tv)
                if abs(der - (fv(tv + hv) - fv(tv))) > 1e-10: bad.append(("미분", f_, tv, hv))
                try:
                    c = mp.findroot(lambda cc: hv*dfv(cc) - (fv(tv + hv) - fv(tv)), tv + hv/2)
                    if not (tv < c < tv + hv): bad.append(("c범위", f_, tv, hv, c))
                except Exception: bad.append(("c없음", f_, tv, hv))
    V.proof(2, "3", bad)
    # [4]
    V.report(2, all(sp.exp(x).diff(x).subs(x, v) > 0 for v in (-5, 0, 7)), "4-1 (e^x)'=e^x>0")
    bad = []
    for tv in [mp.mpf(k)/20 for k in range(1, 400)]:
        mid = mp.e**(tv/(tv + 1)); val = (1 + 1/tv)**tv
        if not (mid < val < mp.e): bad.append(tv)
        S = mp.quad(lambda u: 2/u, [tv, tv + 1])
        if not (2/(tv + 1) < S < 2/tv) or abs(S - 2*mp.log(1 + 1/tv)) > 1e-20: bad.append(("S", tv))
    V.proof(2, "4-2", bad)
    # [5] 우변 미분 = 피적분 (f+g = C e^x, f'=g, g'=f 이므로)
    C = sp.symbols("C")
    u = C*sp.exp(x)
    rhs = sp.sqrt(2)/2*sp.cos(x - sp.pi/4)*u
    V.check(2, "5 d/dx[우변]", sp.simplify(sp.diff(rhs, x) - sp.cos(x)*u), 0)
    f_, g_ = sp.symbols("f_ g_")
    V.check(2, "5 sinx+cosx=√2cos(x-π/4)", sp.simplify(sp.sin(x) + sp.cos(x) - sp.sqrt(2)*sp.cos(x - sp.pi/4)), 0)

if __name__ == "__main__": V.run(globals())
