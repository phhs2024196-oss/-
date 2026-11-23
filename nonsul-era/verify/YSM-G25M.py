import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G25M", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "t": t, "n": n, "k": k})

def p01():
    # 쌍곡선: 2c=14, 2a=10 → b^2=c^2-a^2
    a, c = 5, 7
    b2 = c**2 - a**2
    F, Fp = sp.Matrix([c, 0]), sp.Matrix([-c, 0])
    # P=(px,py) 제1사분면, Q=(q,0) q>0 : ℓ=PQ 가 ∠FPF' 이등분(각의 이등분선 정리: FQ/QF'=PF/PF'), R 는 PF' 위 F'R=10,
    # F'R 을 지름으로 하는 원이 Q 를 지남 ⇔ (R-Q)·(F'-Q)=0
    px, py, q = sp.symbols("px py q", real=True)
    P = sp.Matrix([px, py]); Q = sp.Matrix([q, 0])
    PF, PFp = sp.sqrt((P - F).dot(P - F)), sp.sqrt((P - Fp).dot(P - Fp))
    R = Fp + 10*(P - Fp)/PFp
    eqs = [px**2/a**2 - py**2/b2 - 1,
           (R - Q).dot(Fp - Q),
           (q - (-c))*PF - (c - q)*PFp]          # F'Q : QF = PF' : PF
    sols = sp.nsolve(eqs, [px, py, q], (20, 20, 1), tol=1e-25, maxsteps=200, prec=30)
    pxv, pyv, qv = [sp.nsimplify(float(v), rational=True, tolerance=1e-9) for v in sols]
    V.check(1, "P 좌표(25,24)", [pxv, pyv], [25, 24])
    V.check(1, "Q(1,0)", qv, 1)
    Pv = sp.Matrix([pxv, pyv])
    cos_th = sp.simplify(((Fp - Pv).dot(F - Pv))/(sp.sqrt((Fp - Pv).dot(Fp - Pv))*sp.sqrt((F - Pv).dot(F - Pv))))
    V.check(1, "1-1 cosθ", cos_th, sp.Rational(24, 25))
    # 정적분 값: θ 를 기호로 두고 f 를 넣어 적분 → θ 치환
    th = sp.symbols("th", positive=True)
    f = lambda z: sp.Rational(5)/th*(5*z - 2)**3
    X = sp.symbols("X", real=True)
    fp = sp.diff(f(X), X)
    I1 = sp.integrate(sp.expand(x*sp.cos(x)*fp.subs(X, sp.sin(x))), (x, 0, th))
    I2 = sp.integrate(sp.expand(f(sp.cos(x))), (x, sp.pi/2 - th, sp.pi/2))
    tot = sp.simplify(I1 + I2)
    thv = sp.acos(cos_th)
    V.check(1, "1-1 정적분", sp.N(tot.subs(th, thv), 30), sp.N(E(V.ans(1, "1-1")), 30), 1e-12)
    # 1-2
    Rv = sp.Matrix([1, 6])
    Rv = sp.simplify(Fp + 10*(Pv - Fp)/sp.sqrt((Pv - Fp).dot(Pv - Fp)))
    D = (Rv + 2*F)/3          # RF 를 2:1 로 내분 (R 쪽 1 : F 쪽 2 가 아니라 RD:DF=2:1)
    D = Rv + sp.Rational(2, 3)*(F - Rv)
    d = (Rv - Pv)
    Eref = 2*(Pv + d*((D - Pv).dot(d))/d.dot(d)) - D       # 직선 PR 에 대한 D 의 대칭점
    ka, la = sp.symbols("ka la")
    sol = sp.solve(list(Eref - Pv - ka*(Rv - Pv) - la*(F - Pv)), [ka, la], dict=True)[0]
    V.check(1, "1-2 k", sol[ka], sp.Rational(121, 75))
    V.check(1, "1-2 l", sol[la], sp.Rational(-2, 3))
    V.check(1, "1-2 k+l", sol[ka] + sol[la], E(V.ans(1, "1-2")))
    V.check(1, "1-2 D,E 좌표", [D[0], D[1], Eref[0], Eref[1]], [5, 2, sp.Rational(-43, 25), sp.Rational(274, 25)])
    # 해설 식 (*) : 225k^2-288k-121=0
    kk = sp.symbols("kk")
    a_, b_ = Rv - Pv, F - Pv
    pd = (a_ + 2*b_)/3; pe = kk*a_ - sp.Rational(2, 3)*b_
    V.check(1, "1-2 (*) 이차식", sp.factor(sp.expand((pe.dot(pe) - pd.dot(pd))/ (a_.dot(a_))*225)), sp.factor(225*kk**2 - 288*kk - 121))

def p02():
    Rr = sp.Integer(1); H = sp.sqrt(3)*Rr
    O = sp.Matrix([0, 0, H]); A = sp.Matrix([-Rr, 0, 0]); B = sp.Matrix([Rr, 0, 0])
    C = (O + B)/2
    nrm = B - O
    th = sp.symbols("th", positive=True)
    K = sp.Matrix([-sp.cos(th), sp.sin(th), 0])     # ∠AMK=θ, M 은 원점
    lam = sp.symbols("lam", real=True)
    Ppt = O + lam*(K - O)
    lamv = sp.solve(sp.Eq((Ppt - A).dot(nrm), 0), lam)[0]
    Pv = sp.simplify(Ppt.subs(lam, lamv))
    # OP 위 점이 원뿔 위인지(축에서의 거리 = R*(H-z)/H)
    V.report(2, all(abs(float((sp.sqrt(Pv[0]**2 + Pv[1]**2) - Rr*(H - Pv[2])/H).subs(th, tv))) < 1e-12 for tv in (0.2, 0.9, 1.7, 2.6)), "2-1 P 가 원뿔 위(수치)")
    V.check(2, "2-1 C 가 평면 α 위", (C - A).dot(nrm), 0)
    u = (C - A)
    dist = sp.sqrt(sp.simplify((Pv - A).dot(Pv - A) - ((Pv - A).dot(u))**2/u.dot(u)))
    f = sp.simplify(2*dist)
    g = Rr*th
    Lexp = sp.sympify(V.ans(2, "2-1", "L"), locals={"theta": th})
    for tv in (0.2, 0.5, 1.0, 1.4, 2.0, 2.8):
        V.check(2, f"2-1 L({tv}) 수치", sp.N((f/g).subs(th, tv), 20), sp.N(Lexp.subs(th, tv), 20), 1e-12)
    V.check(2, "2-1 극한", sp.limit(Lexp, th, 0, "+"), E(V.ans(2, "2-1", "lim")))
    # t = MJ : f = 2 t sinθ 이므로 t = 2R/(3-cosθ)
    V.report(2, all(abs(float((f/(2*sp.sin(th)) - 2*Rr/(3 - sp.cos(th))).subs(th, tv))) < 1e-12 for tv in (0.2, 0.9, 1.7, 2.6)), "2-1 t=f/(2 sinθ)=2R/(3-cosθ) (수치)")
    # 2-2: 구의 배치를 단면으로 독립 모델링
    r, s = sp.symbols("r s", positive=True)
    def rad(nv): return Rr*sp.sin(sp.pi/nv)/(1 + sp.sin(sp.pi/nv))
    # 접촉: 중심 반경 ρ=R-r(원기둥 접함), 이웃 구 접함 ρ sin(π/n)=r, 원뿔 모선 접함 → 중심 높이 z0 (거리=r)
    def feasible(nv):
        rv = sp.N(rad(nv), 30)
        rho = Rr - rv
        assert abs(sp.N(rho*sp.sin(sp.pi/nv) - rv, 20)) < 1e-15
        z0 = sp.symbols("z0")
        # 모선: x/R + z/H = 1 → 중심(ρ,z0)에서 거리 = r (원뿔 바깥)
        z0v = sp.solve(sp.Eq((rho*H + z0*Rr - Rr*H)/sp.sqrt(H**2 + Rr**2), rv), z0)[0]
        return z0v + rv < H          # 평면 β(z=H)와 만나지 않음
    cand = [nv for nv in range(3, 41) if nv % 2 == 0 and feasible(nv)]
    best = max(cand, key=lambda nv: float(rad(nv)))
    V.check(2, "2-2 n", best, E(V.ans(2, "2-2", "n")))
    V.report(2, cand[0] == best and 4 not in cand and all(float(rad(a_)) > float(rad(b_)) for a_, b_ in zip(range(3, 30), range(4, 31))), "2-2 r 은 n 에 대해 감소, 가능한 최소 짝수 n")
    V.report(2, [float(sp.sin(sp.pi/nv)) < float(1/sp.sqrt(3)) for nv in range(3, 11)] == [bool(feasible(nv)) for nv in range(3, 11)], "2-2 조건식 ⇔ sin(π/n)<1/√3 (n=3..10)")
    rexp = sp.sympify(V.ans(2, "2-2", "r"), locals={"n": n, "R": 1})
    V.check(2, "2-2 r(n)", sp.N(rexp.subs(n, 6), 20), sp.N(rad(6), 20))

if __name__ == "__main__": V.run(globals())
