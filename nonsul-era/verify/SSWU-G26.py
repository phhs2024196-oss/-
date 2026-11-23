import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("SSWU-G26", "items/SSWU/PAST")
X = sp.Symbol('x'); T = sp.Symbol('t'); N = sp.Symbol('n'); K = sp.Symbol('k'); B = sp.Symbol('b')
E = sp.E

def p01():
    u = sp.Symbol('u', positive=True)
    f1 = E**(X/2) + E**(-X/2)
    d1 = sp.diff(f1, X)
    sol = sp.solve(sp.Eq((u - 1/u)**2, (E - 1/E)**2), u)
    x0 = [sp.simplify(2*sp.log(s)) for s in sol if s > 1][0]
    a = sp.simplify(2*f1.subs(X, x0))
    assert same(-d1.subs(X, x0)**2, -sp.Rational(1, 4)*(E-1/E)**2)
    V.check(1, "1-(1) a (a>4 확인 %s)" % bool(a > 4), a, V.ans(1, "1"))
    f2 = -f1 + a
    area = sp.integrate(f1, (X, 0, x0)) + sp.integrate(f2, (X, x0, 3))
    V.check(1, "1-(2) 넓이", sp.N(area, 30), sp.N(S(V.ans(1, "2")), 30), 1e-12)
    mp.mp.dps = 30
    L = lambda f, lo, hi: mp.quad(sp.lambdify(X, sp.sqrt(1 + sp.diff(f, X)**2), 'mpmath'), [lo, hi])
    per = f1.subs(X, 0) + 3 + f2.subs(X, 3) + L(f1, 0, x0) + L(f2, x0, 3)
    V.check(1, "1-(3) 둘레(수치 적분)", sp.N(per, 25), sp.N(S(V.ans(1, "3")), 25), 1e-12)

def p02():
    ff = sp.Function('f')
    r, y = sp.symbols('r y', positive=True)
    sols = sp.solve(sp.Eq(T**2 + (y - r)**2, r**2), r)
    V.check(2, "2-(2) r(t)", sols[0], S(V.ans(2, "2")).subs(ff(T), y))
    c2, c3 = sp.symbols('c2 c3')
    fs = T + c2*T**2 + c3*T**3
    V.check(2, "2-(3) 극한", sp.limit(((T**2 + fs**2)/(2*fs))/T, T, 0, '+'), V.ans(2, "3"))
    bad = []
    fams = [lambda x: sp.sin(x), lambda x: x - x**2, lambda x: sp.tan(x)/(1+x), lambda x: x - x**3/2]
    for fm in fams:
        for rv in [0.3, 0.5, 0.9, 1.0, 1.4]:
            F = sp.lambdify(X, fm(X), 'math')
            if not F(rv) < rv: continue
            h = lambda x: F(x) - (rv - (rv*rv - x*x)**0.5)
            xs = [rv*i/20000 for i in range(1, 20000)]
            if not any(h(xs[i])*h(xs[i+1]) <= 0 for i in range(len(xs)-1)) and h(xs[0])*h(rv) > 0:
                bad.append((rv,))
    V.proof(2, "2-(1)", bad)

def p03():
    a = sp.Symbol('a'); f = X**3 - 3*X
    ok = []
    for n in range(1, 6):
        g = a*X**n
        lim = sp.limit(sp.integrate(f.subs(X, T)*g.subs(X, T), (T, 0, X))/f**2, X, sp.oo)
        if not lim.has(sp.oo, -sp.oo) and lim != 0:
            ok.append((n, sp.solve(sp.Eq(lim, sp.Rational(1, 3)), a)))
    assert ok == [(2, [2])], ok
    cands = []
    for b in sp.solve(sp.Eq(-3*B - 9, 4), B) + sp.solve(sp.Eq(-B - 1, 4), B):
        g = 2*X**2 + b*X; h = X**2 - g
        mn = min(h.subs(X, v) for v in [1, 3] + [v for v in sp.solve(sp.diff(h, X), X) if 1 <= v <= 3])
        if mn == 4: cands.append(g)
    V.check(3, "3-(1) g", cands[0], V.ans(3, "1"))
    g = cands[0]
    ff = X**3 - 3*X
    val = sp.integrate(sp.apart((g+3)/(ff+2), X), (X, -1, 0))
    V.check(3, "3-(2) 적분", sp.N(val, 30), sp.N(S(V.ans(3, "2")), 30), 1e-12)
    ks = []
    for kk in [sp.Rational(i, 4) for i in range(-8, 17)]:
        rts = set(sp.real_roots(sp.Poly(ff - g - kk*X, X)))
        if len(rts) == 2: ks.append(kk)
    V.check(3, "3-(3) k 집합(1/4 격자) + 판별식", sorted(set(ks) | set(sp.solve(sp.discriminant(X**2-2*X+2-K, X), K)) | set(sp.solve((X**2-2*X+2-K).subs(X, 0), K))), [1, 2])
    V.check(3, "3-(3) k=1 넓이", sp.integrate(sp.Abs(ff-g-X), (X, 0, 1)), V.ans(3, "3", "A1"))
    V.check(3, "3-(3) k=2 넓이", sp.N(sp.integrate(sp.Abs(ff-g-2*X), (X, 0, 2)), 20), sp.N(S(V.ans(3, "3", "A2")), 20), 1e-12)
    V.check(3, "3-(3) k 값", [1, 2], [V.ans(3, "3", "k1"), V.ans(3, "3", "k2")])

def p04():
    import math
    bad = [x_ for x_ in [i*0.0007 for i in range(1, 2244)] if not (math.sin(x_) < x_ < math.tan(x_))]
    bad += [tt for tt in [1e-3, -1e-3, 1e-6, -1e-6] if abs(math.sin(tt)/tt - 1) > 1e-5]
    V.proof(4, "4-(1)", bad)
    for nn in range(2, 9):
        ang = [0.0] + [((3*kk-1)/(3*nn) if kk % 2 else kk/nn)*math.pi for kk in range(1, 2*nn)]
        ar = sum(0.5*math.sin(((ang[(i+1) % (2*nn)] - ang[i]) % (2*math.pi))) for i in range(2*nn))
        Sn = S(V.ans(4, "2", "S")).subs(N, nn)
        assert abs(ar - float(Sn)) < 1e-9, (nn, ar, Sn)
    Sn = S(V.ans(4, "2", "S"))
    V.check(4, "4-(2) 극한 S(2026n)", sp.limit(Sn.subs(N, 2026*N), N, sp.oo), V.ans(4, "2", "L"))
    ts = [i/100 for i in range(-1000, 1001)]
    def theta(tt):
        ax, ay = 0.5, tt
        v1 = (-1-ax, -ay); v2 = (-ax, -ay)
        cr = v1[0]*v2[1]-v1[1]*v2[0]; dt = v1[0]*v2[0]+v1[1]*v2[1]
        return math.atan2(cr, dt)
    th = [theta(tt) for tt in ts]
    assert all(abs(math.tan(theta(tt)) - 4*tt/(3+4*tt*tt)) < 1e-9 for tt in ts)
    V.check(4, "4-(3) 최댓값(수치)", max(abs(h) for h in th), sp.N(S(V.ans(4, "3", "max")), 12), 1e-4)
    fT = 4*T/(3+4*T**2)
    crit = sp.solve(sp.diff(fT, T), T)
    V.check(4, "4-(3) 극값 후 atan", sorted(sp.atan(fT.subs(T, c)) for c in crit), [S(V.ans(4, "3", "min")), S(V.ans(4, "3", "max"))])

if __name__ == "__main__": V.run(globals())
