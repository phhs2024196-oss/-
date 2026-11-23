import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
import mpmath as mp
V = Verifier("KW-G16A", "items/KW/PAST")
SY = lambda s, **kw: sp.sympify(s, locals={"t": t, "x": x, "n": n, **kw})
M = sp.Matrix

def p01():
    # [1] 정수 성분(-4..4)에서 A^2=E, ad=bc+1 인 행렬 전수탐색 → YAML 의 두 행렬과 일치
    found = []
    for a, b, c, d in itertools.product(range(-4, 5), repeat=4):
        A = M([[a, b], [c, d]])
        if A*A == sp.eye(2) and a*d == b*c + 1:
            found.append(A)
    exp = [M(eval(V.ans(1, "1", k))) for k in ("A1", "A2")]
    V.report(1, len(found) == 2 and all(any(f == e for f in found) for e in exp), f"1 C(Z) 해 전수탐색 {len(found)}개 = YAML 2개")
    # [2] f(x)=kx (k>0) 는 (성질 1)과 x>0⇔f(x)>0 을 만족 → W 의 원소 수
    for k_ in (1, 3):
        W = [X for X in range(-60, 61) if k_*X**2 < k_*(4*X - 3)]
        V.check(1, f"2 |W| (f=x*{k_})", len(W), SY(V.ans(1, "2")))
    # [3] g(√3)^2=3, g(√3)>=0 → √3 (항등함수가 (성질1,2) 일대일을 만족하므로 해 존재)
    y = sp.symbols("y", real=True)
    sol = [r for r in sp.solve(y**2 - 3, y) if r >= 0]
    V.check(1, "3 g(√3)", sol[0], SY(V.ans(1, "3")))
    # [4](1) g(m)=m g(1): (성질 1) 만 쓰는 점화 g(m)=g(m-1)+g(1), g(-m)=-g(m)
    bad = []
    for c in range(-6, 7):
        g = {0: 0, 1: c}
        for m in range(2, 31): g[m] = g[m-1] + g[1]
        for m in range(1, 31): g[-m] = -g[m]           # 0=g(0)=g(m)+g(-m)
        bad += [(c, m) for m in range(-30, 31) if g[m] != m*c]
        bad += [(c, m1, m2) for m1 in range(-8, 9) for m2 in range(-8, 9) if g[m1] + g[m2] != g[m1 + m2]]
    V.proof(1, "4-1", bad)
    # [4](2) h^n(A) = [[a, b g(1)^n],[c, d g(1)^n]]
    a, b, c_, d, g1 = sp.symbols("a b c d g1")
    hA = lambda A: M([[A[0, 0], g1*A[0, 1]], [A[1, 0], g1*A[1, 1]]])     # g(m)=m g(1)
    A = M([[a, b], [c_, d]]); cur = A; bad = []
    for nn in range(1, 8):
        cur = hA(cur)
        if sp.simplify(cur - M([[a, b*g1**nn], [c_, d*g1**nn]])) != sp.zeros(2): bad.append(nn)
    V.proof(1, "4-2", bad)
    # [4](3) det h^n(A)≠0 ⇔ det A≠0 (g(1)≠0)
    bad = []
    for g1v in (1, 2, -1, 3, -5):
        for nn in range(1, 5):
            for a_, b_, c2, d_ in itertools.product(range(-2, 3), repeat=4):
                dh = a_*d_*g1v**nn - b_*c2*g1v**nn
                if (dh != 0) != (a_*d_ - b_*c2 != 0): bad.append((g1v, nn, a_, b_, c2, d_))
    V.proof(1, "4-3", bad)

def p02():
    P_ = lambda tt: (sp.cos(tt) + sp.cos(tt)**2, sp.sin(tt) + sp.sin(2*tt)/2)
    Q_ = lambda tt: (3*sp.cos(tt)**2, 3*sp.sin(tt)*sp.cos(tt))
    # [1] x^2+y^2=3x
    qx, qy = Q_(t)
    lhs_, rhs_ = V.ans(2, "1").split("=")
    V.check(2, "1 소거", sp.simplify((qx**2 + qy**2 - SY(rhs_.replace("x", "(qx)"), qx=qx)) ), 0)
    # [2] θ
    px, py = P_(t)
    V.check(2, "2 |OP|^2=(1+cos t)^2", sp.simplify(px**2 + py**2 - (1 + sp.cos(t))**2), 0)
    V.check(2, "2 OP·OQ", sp.simplify(px*qx + py*qy - 3*sp.cos(t)*(1 + sp.cos(t))), 0)
    V.check(2, "2 |OQ|^2", sp.simplify(qx**2 + qy**2 - 9*sp.cos(t)**2), 0)
    for tv, key in ((sp.pi/6, "th1"), (sp.pi/3, "th1"), (sp.Rational(5, 6)*sp.pi, "th2"), (sp.Rational(2, 3)*sp.pi, "th2")):
        u = sp.Matrix(P_(tv)); w = sp.Matrix(Q_(tv))
        cth = sp.simplify(u.dot(w)/(u.norm()*w.norm()))
        V.check(2, f"2 θ(t={tv})", sp.acos(cth), SY(V.ans(2, "2", key)))
    # [3] 교점
    sols = sp.solve([sp.Eq(px, qx), sp.Eq(py, qy)], t, dict=True)
    ts = [s[t] for s in sols if 0 <= s[t] <= sp.pi]
    V.report(2, len(set(ts)) == 1 and sp.simplify(ts[0] - sp.pi/3) == 0, f"3 교점의 t = {set(ts)}")
    cx, cy = P_(sp.pi/3)
    V.check(2, "3-1 x", sp.simplify(cx), SY(V.ans(2, "3-1", "x")))
    V.check(2, "3-1 y", sp.simplify(cy), SY(V.ans(2, "3-1", "y")))
    V.check(2, "3-2 β", sp.atan(sp.simplify(cy/cx)), SY(V.ans(2, "3-2")))
    # [4]
    V.check(2, "4-1 dy/dx", sp.simplify((sp.diff(py, t)/sp.diff(px, t)).subs(t, sp.pi/3)), SY(V.ans(2, "4-1")))
    slope = sp.simplify((sp.diff(qy, t)/sp.diff(qx, t)).subs(t, sp.pi/3))
    line = slope*(x - cx) + cy
    ly, lr = V.ans(2, "4-2").split("=")
    V.check(2, "4-2 접선", sp.expand(sp.simplify(line)), sp.expand(SY(lr)))
    # 음함수 미분 기울기와 일치
    V.check(2, "4-2 음함수 기울기", sp.simplify((3 - 2*cx)/(2*cy)), slope)
    # [5] 곡선 길이
    f5 = sp.simplify(sp.diff(px, t)**2 + sp.diff(py, t)**2 - 4*sp.cos(t/2)**2)
    V.check(2, "5 P 속력^2", f5, 0)
    V.check(2, "5 Q 속력^2", sp.simplify(sp.diff(qx, t)**2 + sp.diff(qy, t)**2 - 9), 0)
    s_ = sp.symbols("s", positive=True)
    D = sp.integrate(2*sp.cos(t/2), (t, 0, s_)); L = sp.integrate(3, (t, 0, s_))
    mp.mp.dps = 30
    for sv in (0.7, 2.0, 3.0):
        qd = mp.quad(lambda tt: mp.sqrt((-mp.sin(tt) - mp.sin(2*tt))**2 + (mp.cos(tt) + mp.cos(2*tt))**2), [0, sv])
        V.check(2, f"5 D({sv}) 수치적분", qd, sp.N(D.subs(s_, sv), 25), tol=1e-12)
    V.check(2, "5 극한", sp.limit(D/L, s_, 0, "+"), SY(V.ans(2, "5")))

if __name__ == "__main__": V.run(globals())
