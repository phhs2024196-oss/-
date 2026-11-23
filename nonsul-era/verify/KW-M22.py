import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-M22", "items/KW/PAST")
R_ = sp.Rational

def p01():
    # [2] 두 조건: 수치 격자 탐색
    import math
    def cond_a(k):
        return all(math.cos(t_)**2 + (k+2)*math.sin(t_) - 2*k - 1 > 1e-12 for t_ in [i*2*math.pi/2000 for i in range(2001)])
    def cond_b(k):
        # sqrt(2x+1) - 2x + k = 0 의 서로 다른 실근 수 (x>=-1/2): 제곱 후 근 걸러내기
        a_, b_, c_ = 4, -2*(2*k+1), k*k-1
        D = b_*b_ - 4*a_*c_
        if D < 0: return 0
        roots = {(-b_ + s*math.sqrt(D))/(2*a_) for s in (1, -1)}
        ok = [r for r in roots if r >= -0.5 - 1e-12 and 2*r - k >= -1e-12]
        return len(set(round(r, 12) for r in ok))
    good = [i/400 for i in range(-800, 200) if cond_a(i/400) and cond_b(i/400) == 2]
    lo, hi = sp.sympify(V.ans(1, "2", "lo")), sp.sympify(V.ans(1, "2", "hi").split("(")[0])
    V.report(1, abs(min(good) - float(lo)) < 0.01 + 1/400 and abs(max(good) - float(hi)) < 0.01 + 1/400 and all(float(lo) < g < float(hi) for g in good), f"2 k 범위(수치 격자): {min(good)}<k<{max(good)} (grid 1/400) vs ({lo},{hi})")
    # [3](1)
    A = sp.Matrix([sp.sqrt(3), 1]); B = sp.Matrix([-1, -sp.sqrt(3)])
    dot = A.dot(B); th = sp.acos(dot/(A.norm()*B.norm()))
    V.check(1, "3-1 내적", dot, -2*sp.sqrt(3))
    V.check(1, "3-1 각(도)", sp.simplify(th*180/sp.pi), V.ans(1, "3-1", "angle"))
    V.check(1, "3-1 넓이", sp.Rational(1, 2)*abs(A[0]*B[1]-A[1]*B[0]), V.ans(1, "3-1", "area"))
    # [3](2) 외접원 위 점 P 의 매개변수 탐색으로 사각형 OAPB 넓이 최대
    ab = sp.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    R = sp.simplify(ab/(2*sp.sin(150*sp.pi/180)))
    # 외심: O,A,B 를 지나는 원
    cx, cy = sp.symbols("cx cy", real=True)
    sol = sp.solve([cx**2 + cy**2 - R**2, (cx-A[0])**2 + (cy-A[1])**2 - R**2, (cx-B[0])**2 + (cy-B[1])**2 - R**2], [cx, cy], dict=True)
    import math
    best = 0.0
    Rf = float(R)
    for s_ in sol:
        c0 = (float(s_[cx]), float(s_[cy]))
        Af = (float(A[0]), float(A[1])); Bf = (float(B[0]), float(B[1]))
        for i in range(0, 7200):
            q = math.pi*i/3600
            P = (c0[0] + Rf*math.cos(q), c0[1] + Rf*math.sin(q))
            pts = [(0, 0), Af, P, Bf]
            area = abs(sum(pts[j][0]*pts[(j+1) % 4][1] - pts[(j+1) % 4][0]*pts[j][1] for j in range(4)))/2
            best = max(best, area)
    V.check(1, "3-2 최대 넓이(수치 최대)", best, sp.N(sp.sympify(V.ans(1, "3-2"))), tol=2e-3)
    V.check(1, "3-2 R", sp.simplify(R), sp.sqrt(2)*(sp.sqrt(3)+1))
    # [4]
    h = x*sp.log(x) - x
    V.check(1, "4 alpha", sp.solve(sp.diff(h, x), x)[0], V.ans(1, "4", "alpha"))
    be = sp.solve(sp.Eq(sp.diff(h, x), 1), x)[0]
    V.check(1, "4 beta", be, sp.E if V.ans(1, "4", "beta") == "e" else V.ans(1, "4", "beta"))
    al = 1
    hf = sp.lambdify(x, h, "math")
    bef = float(be)
    ratios = [(hf(1 + (bef-1)*i/4000) + 2)/(1 + (bef-1)*i/4000) for i in range(0, 4001)]
    V.check(1, "4 p 최댓값", min(ratios), sp.N(sp.log(2)), tol=1e-6)
    V.check(1, "4 q 최솟값", max(ratios), V.ans(1, "4", "qmin"), tol=1e-9)
    V.check(1, "4 ln2 값", sp.N(sp.sympify(V.ans(1, "4", "pmax"), locals={"ln": sp.log})), min(ratios), tol=1e-6)

def p02():
    f = x**3*sp.exp(-x)
    # [1](1) 홀함수
    g = x**2022*f*sp.exp(x**2 + x)
    V.check(2, "1-1 g 홀함수", sp.simplify(g.subs(x, -x) + g), 0)
    gn = sp.lambdify(x, g, "mpmath")
    import mpmath as mp
    mp.mp.dps = 40
    I1 = mp.quad(gn, [0, mp.e]); I2 = mp.quad(gn, [-mp.e, 0])
    V.report(2, abs((I1 + I2)/I1) < 1e-20 and sp.sympify(V.ans(2, "1-1")) == 0, "1-1 정적분 상쇄(상대오차<1e-20)")
    # [1](2)
    crit = sorted(sp.solve(sp.diff(f, x), x))
    V.check(2, "1-2 임계점", crit, [0, 3])
    V.check(2, "1-2 최댓값", f.subs(x, 3), sp.sympify(V.ans(2, "1-2"), locals={"exp": sp.exp}))
    V.report(2, sp.N(f.subs(x, 3)) > 1.34 and sp.N(f.subs(x, 3)) < 1.345, "1-2 f(3)≈1.344<3")
    # [1](3): 합성함수의 극값 위치 (수치 부호 변화)
    ff = lambda v: f.subs(x, f.subs(x, v))
    dff = sp.diff(f.subs(x, f), x)
    dffn = sp.lambdify(x, dff, "math")
    ext = []
    import math
    pts = [i/50 for i in range(-100, 700)]
    vals = [dffn(p_) for p_ in pts]
    for i in range(1, len(pts)-1):
        if vals[i-1]*vals[i+1] < 0 and abs(vals[i]) < 1e3: ext.append(round(float(pts[i]), 2))
    # 극값 후보: 부호 변화 점 (x=3 근처만 나와야 함)
    V.report(2, ext and all(abs(e - 3) < 0.05 for e in ext), f"1-3 (f∘f)' 부호 변화 위치 {ext[:4]} (x=3 만)")
    V.report(2, int(V.ans(2, "1-3")) == 3, "1-3 YAML 답 3")
    # [2](1)
    sols = [-sp.pi/2, 3*sp.pi/2]
    V.check(2, "2-1 합", sum(sols), V.ans(2, "2-1", "a"))
    V.check(2, "2-1 곱", sols[0]*sols[1], sp.sympify(V.ans(2, "2-1", "b")))
    # 수치로 -2pi..2pi 의 해가 이 둘뿐인지
    import math
    cnt = [i for i in range(-4000, 4001) if abs(math.sin(i*math.pi/2000)**2021 + 1 + math.sqrt(2)*abs(math.cos(i*math.pi/2000))) < 1e-6]
    V.report(2, set(round(i/2000, 3) for i in cnt) <= {-0.5, 1.5}, "2-1 격자에서 해는 -pi/2, 3pi/2 부근뿐")
    # [2](2)
    n_ = sp.symbols("n_", integer=True, positive=True)
    S = lambda m: sp.pi*m**2 - sp.Rational(3, 4)*sp.pi*m
    an = sp.sympify(V.ans(2, "2-2", "an"), locals={"n": n_})
    V.check(2, "2-2 a1", S(1), an.subs(n_, 1))
    V.check(2, "2-2 an=S_n-S_(n-1)", sp.simplify(S(n_) - S(n_-1) - an), 0)
    # [3] F, f
    F = sp.Function("F")
    Fx = x*sp.exp(x) + 1 - sp.exp(x)         # ODE F' = F - 1 + e^x, F(0)=0 의 해
    fx = sp.diff(Fx, x)
    V.check(2, "3 조건 (a)", sp.simplify(Fx - (fx + 1 - sp.exp(x))), 0)
    V.check(2, "3 조건 (b) F(1)=1", Fx.subs(x, 1), 1)
    V.check(2, "3-1 ∫F", integral(Fx, x, 0, 1), 3 - sp.E)
    V.check(2, "3-1 ∫xf", integral(x*fx, x, 0, 1), sp.sympify(V.ans(2, "3-1"), locals={"e": sp.E}))
    I2 = integral(Fx*(sp.exp(x) + Fx), x, 0, 1)
    V.check(2, "3-2 적분값", I2, sp.sympify(V.ans(2, "3-2", "integral"), locals={"exp": sp.exp}))
    V.check(2, "3-2 최대 정수 n", sp.floor(sp.N(I2)) if sp.N(I2) != sp.floor(sp.N(I2)) else sp.floor(sp.N(I2))-1, V.ans(2, "3-2", "n"))

if __name__ == "__main__": V.run(globals())
