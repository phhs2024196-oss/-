"""SSWU-G25 (성신여대 2025학년도 논술 자연계열 1~4) 검산 — 문제 조건에서 독립 재계산, 기대값은 V.ans()."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("SSWU-G25", "items/SSWU/PAST")
mp.mp.dps = 30
LOC = {"x": x, "n": n, "k": k, "e": sp.E}
def E(s): return sp.sympify(s, locals=LOC)
def kv(no, sub):
    return {p.split("=", 1)[0].strip(): p.split("=", 1)[1] for p in V.ans(no, sub).replace(" ", "").split(";") if "=" in p}

def p01():
    c = sp.symbols("c", real=True); X = sp.symbols("X", real=True)
    cs = sp.solve(sp.Eq(0, -sp.exp(-c) * (n - c) + sp.exp(-c)), c)[0]
    line = -sp.exp(-cs) * (x - cs) + sp.exp(-cs)
    d = kv(1, "1")
    V.check(1, "1-1 c", cs, E(d["Px"])); V.check(1, "1-1 Py", sp.exp(-cs), E(d["Py"]))
    V.check(1, "1-1 line", sp.simplify(line - E(d["l"])), 0)
    # (2) 구간: n=1,2,3 에서 수치 적분(접선이 x축 위인 부분만 제1사분면)
    for nn in (1, 2, 3):
        cc = nn - 1; m = -mp.e ** (-cc); f = lambda t: mp.e ** (-t); L = lambda t: m * (t - cc) + f(cc)
        # 제1사분면: 곡선과 접선 사이 + y=0 위쪽, x in [cc? ... 접선은 x=n 에서 0. 영역은 [n-1,2n] 중 접선 위(y>=0)와 곡선 아래
        a = mp.quad(lambda t: f(t) - max(L(t), 0), [cc, nn, 2 * nn])
        V.check(2, f"1-2 a_n n={nn}", a, sp.exp(1 - nn) / 2 - sp.exp(-2 * nn), 1e-20)
    an = sp.exp(1 - n) / 2 - sp.exp(-2 * n)
    V.check(1, "1-2 limit", sp.limit(sp.exp(n) * an, n, sp.oo), E(V.ans(1, "2")))
    # (3) 법선: 기울기 e^(n-1), P 지나고 x절편, 삼각형 면적
    Px, Py = cs, sp.exp(-cs)
    xint = sp.solve(sp.Eq(0, sp.exp(n - 1) * (x - Px) + Py), x)[0]
    b = sp.simplify((Px - xint) * Py / 2)
    V.check(1, "1-3 b_n", b, sp.exp(3 - 3 * n) / 2)
    V.check(1, "1-3 sum", sp.summation(b, (n, 1, sp.oo)), E(V.ans(1, "3")))

def p02():
    # (1)
    al, be = sp.atan(2 * sp.sqrt(3)), sp.atan(sp.sqrt(3) / 7)
    P = sp.solve([sp.Eq(y, 2 * sp.sqrt(3) * (x - 1)), sp.Eq(y, sp.sqrt(3) / 7 * x)], [x, y])
    Pp = sp.Matrix([P[x], P[y]]); O = sp.Matrix([0, 0]); Q = sp.Matrix([1, 0])
    u, w = O - Pp, Q - Pp
    cosv = sp.simplify(u.dot(w) / (u.norm() * w.norm()))
    V.check(2, "2-1 cos", cosv, V.ans(2, "1"))
    # (2) R: 제3사분면, y=mx 위, RP=2
    t = sp.symbols("t", real=True); R = sp.Matrix([t, sp.sqrt(3) / 7 * t])
    sol = [s for s in sp.solve(sp.Eq((R - Pp).dot(R - Pp), 4), t) if s < 0]
    assert len(sol) == 1
    Rv = R.subs(t, sol[0]); rq2 = sp.simplify((Rv - Q).dot(Rv - Q))
    V.check(2, "2-2 RQ^2", rq2, V.ans(2, "2"))
    # (3) f(θ)=OP 수치 극한
    for kk in (2, 3, sp.Rational(5, 2), 7):
        def f(th):
            a, m = mp.tan(kk * th), mp.tan(th)
            xp = a / (a - m); return xp / mp.cos(th)  # P=(xp, m xp), OP = xp*sqrt(1+m^2)
        V.check(2, f"2-3 k={kk}", f(mp.mpf("1e-12")), E(V.ans(2, "3")).subs(k, kk), 1e-9)

def p03():
    a = sp.symbols("a", positive=True)
    f = x**2 * (x - 3 * a + 1); g = -a * x * (x + 3)
    sol = [s for s in sp.solve(sp.Eq(sp.integrate(g - f, (x, -2, 0)), 58), a) if s > 4]
    V.check(3, "3-1 a", sol[0], V.ans(3, "1"))
    # [-2,0] 에서 g>=f 확인
    V.report(3, all(((g - f).subs(a, sol[0]).subs(x, v)) >= 0 for v in [-2, -1.5, -1, -0.5, 0]), "g>=f on [-2,0]")
    A = sol[0]; F = f.subs(a, A); G = g.subs(a, A)
    b = sp.symbols("b", real=True)
    cand = []
    for xs in sp.solve(sp.Eq(sp.diff(F, x), sp.diff(G, x)), x):
        bs = sp.solve(sp.Eq(F.subs(x, xs), (G + b).subs(x, xs)), b)[0]
        cand.append((xs, bs))
    bpos = [bs for xs, bs in cand if bs > 0]; assert len(bpos) == 1
    d = kv(3, "2")
    V.check(3, "3-2 b", bpos[0], E(d["b"]))
    h = sp.expand(F - (G + bpos[0])); roots = sp.roots(sp.Poly(h, x))
    lo, hi = min(roots), max(roots)
    area = sp.integrate((G + bpos[0]) - F, (x, lo, hi))
    V.check(3, "3-2 area", area, E(d["area"]))
    # (3) 정의로 검사: h 의 좌·우 미분계수 (c 후보와 비후보)
    mp.mp.dps = 40
    Ff = sp.lambdify(x, F, "mpmath")
    def hh(cc, xv): return Ff(xv) if xv < cc else Ff(2 * cc - xv)
    def jump(cc, eps=mp.mpf("1e-12")):
        l = (hh(cc, cc - eps) - hh(cc, cc)) / (-eps); r = (hh(cc, cc + eps) - hh(cc, cc)) / eps
        return abs(l - r)
    ans = [E(s) for s in V.ans(3, "3").split(",")]
    V.report(3, all(jump(mp.mpf(str(sp.N(c, 30)))) < 1e-8 for c in ans), "3-3 정답 c 에서 미분가능")
    V.report(3, all(jump(mp.mpf(v)) > 1e-3 for v in (1, 2, 5, 12, -1)), "3-3 다른 c 는 불가")
    fp = sp.diff(F, x); V.check(3, "3-3 f'(c)=0 근", sorted(sp.solve(fp, x)), sorted(ans))

def p04():
    n_ = sp.symbols("n", positive=True); kk = sp.symbols("kk", positive=True)
    f = kk / (x - n_)
    P = sp.Matrix([2 * n_, f.subs(x, 2 * n_)]); N = sp.Matrix([n_, 0])
    Q = 2 * N - P  # PN 을 2:1 로 외분: (2N - P)/(2-1)
    area = sp.simplify(sp.Rational(1, 2) * abs(P[0] - Q[0]) * abs(P[1] - Q[1]))
    ks = sp.solve(sp.Eq(area, 2 / n_**2), kk)[0]
    V.check(4, "4-1 f", f.subs(kk, ks), E(V.ans(4, "1")).subs(n, n_))
    # (2) n=1
    fx = lambda t: 1 / (t - 1); sym = lambda t: 1 / t + 1
    top = lambda t: min(fx(t), sym(t)); 
    mp.mp.dps = 30
    ar = mp.quad(lambda t: top(t) - (t - 1), [1, (1 + mp.sqrt(5)) / 2, 2])
    V.check(4, "4-2 area", ar, E(V.ans(4, "2")), 1e-20)
    # 대칭곡선: y=1/(x-1) 위의 점 (t, 1/(t-1)) 의 대칭점 (1/(t-1), t) 가 y=1/x+1 위
    V.report(4, all(abs(sym(fx(t)) - t) < 1e-20 for t in (mp.mpf(1.5), mp.mpf(3))), "4-2 대칭곡선")
    # (3) 정확한 유리수 격자로 최소 n
    from fractions import Fraction as Fr
    def ok(nn):
        F = lambda t: Fr(1, nn * nn) / (t - nn)
        for i in range(0, 2001):
            t = Fr(nn + 2) + Fr(i, 2000)
            if (nn + 1) ** 2 * F(t) > nn**2 * F(t - 1): return False
        return True
    mn = next(nn for nn in range(1, 30) if ok(nn))
    V.check(4, "4-3 min n", mn, V.ans(4, "3"))
    V.report(4, not ok(4) and ok(5) and ok(6), "4-3 n=4 불가, 5·6 가능")

if __name__ == "__main__": V.run(globals())
