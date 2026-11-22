"""KAU-G24S (한국항공대 2024 이학계열) 검산 — 설정에서 독립 재계산하고 YAML 정답(V.ans)과 비교."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import math
import mpmath as mp
V = Verifier("KAU-G24S", "items/KAU/PAST")
a_, b_ = sp.symbols("a b", real=True)


def p01():
    # 1-1: z=a+bi, b=-sqrt(3)a, z^2 = -conj(z)/2 를 실수부·허수부 방정식으로 풀고 p=-(z+z̄), q=z z̄
    z = a_ + b_ * sp.I
    eq = sp.expand(z**2 + sp.conjugate(z) / 2)
    sols = sp.solve([sp.re(eq).subs(b_, -sp.sqrt(3) * a_), sp.im(eq).subs(b_, -sp.sqrt(3) * a_)], a_, dict=True)
    av = [s_[a_] for s_ in sols if s_[a_] != 0][0]
    zz = z.subs({b_: -sp.sqrt(3) * a_}).subs(a_, av)
    V.check(1, "1-1 p", sp.simplify(-(zz + sp.conjugate(zz))), V.ans(1, "1", "p"))
    V.check(1, "1-1 q", sp.simplify(zz * sp.conjugate(zz)), V.ans(1, "1", "q"))
    # 1-2: h(k) = ∫_{-5}^{4} f(g(x)) dx 를 수치 적분(f(u)=max(u,0)) 으로 k 격자 전수
    def h(k):
        g = lambda u: max(-(u - k) ** 2 + 4, 0.0)
        return mp.quad(g, [-5, max(-5, min(4, k - 2)), max(-5, min(4, k + 2)), 4])
    lo, hi = [S(v) for v in V.ans(1, "2", "range").split("<=k<=")]
    best = mp.mpf(32) / 3
    mx, bad = 0, []
    for i in range(-80, 81):
        k = i / 10
        hv = h(k)
        mx = max(mx, hv)
        inr = float(lo) - 1e-9 <= k <= float(hi) + 1e-9
        if (abs(hv - best) < 1e-9) != inr:
            bad.append(k)
    V.proof(1, "1-2 (1) 최대가 되는 k 범위 [수치확인 k=-8..8 간격 0.1]", bad)
    V.check(1, "1-2 (2) 최댓값", sp.nsimplify(float(mx), rational=True, tolerance=1e-9), V.ans(1, "2", "max"))
    # 1-3: (1) b3-a3 기호 계산, (2)(3) 수치 반례 탐색
    h_ = sp.symbols("h", positive=True)
    a3 = 1 + 2 * h_; b3 = (1 + h_) ** 2
    V.check(1, "1-3 (1) b3-a3", sp.expand(b3 - a3), sp.sympify(V.ans(1, "3", "b3-a3"), locals={"h": h_}))
    bad = []
    for hh in [sp.Rational(i, 20) for i in range(1, 100)]:
        an, bn = sp.Integer(1), sp.Integer(1)
        for n in range(1, 60):
            if n >= 3 and not bn > an:
                bad.append((hh, n))
            an, bn = an + hh, bn * (1 + hh)
    V.proof(1, "1-3 (3) n>=3 에서 b_n>a_n [수치확인 h=0.05..4.95, n<=59]", bad)
    # 1-4: R_n 을 직접 구성: R_n 한 변 s_n, 왼쪽 아래 (X_n,0); 원 O_n 은 오른쪽 위 1/4 정사각형에 내접
    X, s = sp.Integer(0), sp.Integer(4)
    cs, rs = {}, {}
    for n in range(1, 13):
        cs[n] = (X + sp.Rational(3, 4) * s, sp.Rational(3, 4) * s)
        rs[n] = s / 4
        X, s = X + s, s / 2
    nn = sp.Symbol("n")
    rexp = S(V.ans(1, "4", "r"), locals={"n": nn})
    V.report(1, all(same(rexp.subs(nn, n), rs[n]) for n in rs), "1-4 (1) r_n = 구성한 반지름과 n=1..12 일치")
    inc = S(V.ans(1, "4", "x").replace("x(n-1)", "0"), locals={"n": nn})
    V.report(1, all(same(cs[n][0] - cs[n - 1][0], inc.subs(nn, n)) for n in range(2, 13)), "1-4 (2) x_n - x_(n-1) 일치 n=2..12")
    slope = (cs[10][1] - cs[9][1]) / (cs[10][0] - cs[9][0])
    V.check(1, "1-4 (3) 기울기", slope, V.ans(1, "4", "slope"))
    V.report(1, len({(cs[n + 1][1] - cs[n][1]) / (cs[n + 1][0] - cs[n][0]) for n in range(1, 12)}) == 1, "1-4 모든 중심이 한 직선 위(기울기 일정)")


def p02():
    # 2-1
    f = -x**3 + 6 * x
    cp = sorted(sp.solve(sp.diff(f, x), x), key=float)
    xa, xb = cp                      # f'' 부호로 극소·극대 판정
    assert sp.diff(f, x, 2).subs(x, xa) > 0 > sp.diff(f, x, 2).subs(x, xb)
    m = sp.simplify((f.subs(x, xb) - f.subs(x, xa)) / (xb - xa))
    V.check(2, "2-1 (1) m", m, V.ans(2, "1", "m"))
    lo_s, hi_s = V.ans(2, "1", "range").split("<k<")
    lo, hi = S(lo_s), S(hi_s)
    bad = []
    for i in range(-40, 41):
        kv = sp.Rational(i, 4)
        nroot = len(sp.Poly(f - m * x - kv, x).nroots(n=30))
        real = [r for r in sp.Poly(f - m * x - kv, x).nroots(n=30) if abs(sp.im(r)) < 1e-12]
        distinct = len({round(float(sp.re(r)), 8) for r in real})
        if (distinct == 3) != (lo < kv < hi):
            bad.append(kv)
    V.proof(2, "2-1 (2) 교점 3개인 k 범위 [수치확인 k=-10..10 간격 0.25]", bad)
    # 2-2
    a = sp.Symbol("a", real=True)
    def two_pos(av):
        D = 25 * (av + 1) ** 2 - 36 * av**2
        return D > 0 and 5 * (av + 1) / 3 > 0 and av**2 > 0
    r1 = V.ans(2, "2", "range1").split("<a<"); r2 = V.ans(2, "2", "range2").split("<a<")
    rngs = [(S(r1[0]), S(r1[1])), (S(r2[0]), S(r2[1]))]
    bad = [av for av in [sp.Rational(i, 20) for i in range(-60, 140)]
           if two_pos(av) != any(l < av < h for l, h in rngs)]
    V.proof(2, "2-2 (1) 서로 다른 두 양근 조건 [수치확인 a=-3..7 간격 0.05]", bad)
    al = sp.Symbol("al", positive=True)
    cand = sp.solve([al + 9 * al - sp.Rational(5, 3) * (a + 1), 9 * al**2 - a**2], [al, a], dict=True)
    avals = sorted({c_[a] for c_ in cand if two_pos(c_[a])}, key=float)
    # 직접 확인: 해당 a 에서 -√β,-√α,√α,√β 가 등차수열인지
    for av in avals:
        rts = sorted(sp.solve(3 * x**2 - 5 * (av + 1) * x + 3 * av**2, x), key=float)
        q = [-sp.sqrt(rts[1]), -sp.sqrt(rts[0]), sp.sqrt(rts[0]), sp.sqrt(rts[1])]
        V.report(2, same(q[1] - q[0], q[2] - q[1]) and same(q[2] - q[1], q[3] - q[2]), f"2-2 (2) a={av} 에서 등차수열")
    yv = sorted((S(v) for v in V.ans(2, "2", "avals").split(",")), key=float)
    V.report(2, len(yv) == len(avals) and all(same(p, q_) for p, q_ in zip(yv, avals)), f"2-2 (2) a 값 {avals} / YAML {yv}")
    # 2-3
    hk = V.ans(2, "3", "h")
    kk = sp.Symbol("k", positive=True)
    for kv in (0.25, 1, 2, 5, 9):
        A = (kv**2, kv); B = (-math.sqrt(kv), kv)
        assert abs(math.sqrt(A[0]) - kv) < 1e-12 and abs(B[0] ** 2 - kv) < 1e-12
        cosv = (A[0] * B[0] + A[1] * B[1]) / (math.hypot(*A) * math.hypot(*B))
        exp = float(S(hk).subs(sp.Symbol("k"), kv))
        V.report(2, abs(cosv - exp) < 1e-12, f"2-3 (1) k={kv} cos={cosv:.10f} / YAML {exp:.10f}")
    expr = sp.sqrt(kk) * S(hk, locals={"k": kk})
    V.check(2, "2-3 (2) 극한", sp.limit(expr, kk, sp.oo), V.ans(2, "3", "limit"))
    # 2-4
    al, be = sp.symbols("alpha beta", real=True)
    fx = x**2 - (al + be) * x + al * be
    cond = sp.solve(sp.Eq(sp.integrate(fx, (x, 0, 1)), sp.Rational(1, 3)), be)[0]       # beta = alpha/(2alpha-1)
    Sa = sp.simplify(sp.integrate((2 * al - 1) * fx.subs(be, cond), (x, -al, al)))
    crit = [c for c in sp.solve(sp.diff(Sa, al), al) if c != sp.Rational(1, 2)]
    Sd = sp.diff(Sa, al)
    mins = [c for c in crit if Sd.subs(al, c - sp.Rational(1, 100)) < 0 < Sd.subs(al, c + sp.Rational(1, 100))]
    assert len(mins) == 1
    V.check(2, "2-4 alpha1", mins[0], V.ans(2, "4", "alpha1"))
    V.check(2, "2-4 m", sp.simplify(Sa.subs(al, mins[0])), V.ans(2, "4", "m"))
    # 전역 최솟값 확인(수치)
    vals = [float(Sa.subs(al, sp.Rational(i, 50))) for i in range(-200, 201) if i != 25]
    V.report(2, min(vals) >= float(S(V.ans(2, "4", "m"))) - 1e-9, "2-4 α=-4..4 격자에서 S 가 m 이상 [수치확인]")


if __name__ == "__main__":
    V.run(globals())
