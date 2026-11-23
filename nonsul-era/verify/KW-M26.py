"""광운대 2026학년도 모의논술 자연계열(KW-M26) 검산 — 기대값은 V.ans() 로만 읽는다."""
import sys, pathlib, itertools, math
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-M26", "items/KW/PAST")


def p01():
    # [1] 세 꼭짓점과 이등변 조건을 모두 따져 m>0 의 해를 구한다
    m = sp.symbols("m", positive=True)
    O = sp.Matrix([0, 0])
    P = sp.Matrix(list(sp.solve([y - 3*x, y - (m*x + 2)], [x, y]).values()))
    Pd = sp.solve([y - 3*x, y - (m*x + 2)], [x, y]); Qd = sp.solve([y + x/3, y - (m*x + 2)], [x, y])
    P = sp.Matrix([Pd[x], Pd[y]]); Q = sp.Matrix([Qd[x], Qd[y]])
    s_op, s_oq, s_pq = (P - O).norm()**2, (Q - O).norm()**2, (P - Q).norm()**2
    ms = set()
    for lhs, rhs in [(s_op, s_oq), (s_op, s_pq), (s_oq, s_pq)]:
        for r in sp.solve(sp.Eq(sp.simplify(lhs), sp.simplify(rhs)), m):
            if r.is_positive: ms.add(sp.nsimplify(r))
    V.report(1, ms == {sp.Rational(1, 2)}, f"1 이등변 조건의 m>0 해 {ms}")
    area = sp.Abs(P[0]*Q[1] - P[1]*Q[0]) / 2
    V.check(1, "1 넓이", sp.simplify(area.subs(m, sp.Rational(1, 2))), V.ans(1, "1"))
    # [2] 정확한 유리수 합으로 a_n 확인 (n=1..9)
    ok = True
    nn = sp.symbols("nn", integer=True, positive=True)
    for N in range(1, 10):
        lhs = sum(Fraction(4*kk*kk + 2*kk - 1, kk + 1) * math.comb(N, kk) for kk in range(N + 1))
        a_n = S(V.ans(1, "2")).subs(nn, N) if False else sp.sympify(V.ans(1, "2"), locals={"n": n}).subs(n, N)
        ok &= (sp.Rational(lhs.numerator, lhs.denominator) == sp.Rational(a_n, N + 1))
    V.report(1, ok, "2 a_n/(n+1)=Σ… (n=1..9 유리수 정확 비교)")
    # [3] a 를 수치로 찾고(최댓값=2 가 되는 a>0, 이분법) 그 a 에서 최솟값
    xs = [2*math.pi*i/20000 for i in range(20000)]
    def fmax(a): return max(math.sin(v)**2 + 2*a*math.cos(v) - 1 for v in xs)
    lo, hi = 1e-6, 10.0
    for _ in range(60):
        mid = (lo + hi)/2
        if fmax(mid) < 2: lo = mid
        else: hi = mid
    a_sol = (lo + hi)/2
    fmin = min(math.sin(v)**2 + 2*a_sol*math.cos(v) - 1 for v in xs)
    V.check(1, "3 최솟값", fmin, sp.N(S(V.ans(1, "3"))), 1e-6)
    # 형태 확인: f(x)=cos²(x+π/2)−2a cos(x+π)−1 을 원식 그대로 평가해서도 같은 값
    a_s = sp.Rational(3, 2)
    orig = lambda v: math.cos(v + math.pi/2)**2 - 2*float(a_s)*math.cos(v + math.pi) - 1
    V.check(1, "3 원식 (a=3/2) 최댓값·최솟값", (max(orig(v) for v in xs), min(orig(v) for v in xs)), (2, sp.N(S(V.ans(1, "3")))), 1e-6)
    # [4](1) 실수 세제곱근을 쓰는 분기식으로 극값·변곡점
    fpos = sp.Rational(3, 5)*x**sp.Rational(5, 3) - 3*x**sp.Rational(2, 3)            # x>0
    fneg = -sp.Rational(3, 5)*(-x)**sp.Rational(5, 3) - 3*(-x)**sp.Rational(2, 3)    # x<0 (실수 세제곱근)
    xp = sp.symbols("xp", positive=True); xn = sp.symbols("xn", negative=True)
    cp = sp.solve(sp.diff(fpos.subs(x, xp), xp), xp)
    infl_pos = sp.solve(sp.diff(fpos.subs(x, xp), xp, 2), xp)
    infl_neg = sp.solve(sp.diff(fneg.subs(x, xn), xn, 2), xn)
    cn = sp.solve(sp.diff(fneg.subs(x, xn), xn), xn)
    cn = [r for r in cn if abs(sp.im(sp.N(r))) < 1e-9 and sp.re(sp.N(r)) < 0]
    V.report(1, cp == [2] and not cn and not infl_pos and infl_neg == [-1], f"4-1 실수 임계점 {cp}, {cn} / 변곡 {infl_pos}, {infl_neg}")
    fmin_v = sp.simplify(fpos.subs(x, 2)); finfl = sp.simplify(fneg.subs(x, -1))
    V.check(1, "4-1 극솟값", fmin_v, V.ans(1, "4-1", "min"))
    V.check(1, "4-1 변곡점 y", finfl, V.ans(1, "4-1", "infly")); V.check(1, "4-1 변곡점 x", infl_neg[0], V.ans(1, "4-1", "inflx"))
    # x=0 에서 극대: 0 근방 격자에서 f<=f(0)=0
    def fnum(v): return float(fpos.subs(x, v)) if v > 0 else float(fneg.subs(x, v)) if v < 0 else 0.0
    V.report(1, all(fnum(i/1000) <= 1e-12 for i in range(-300, 301) if i != 0), "4-1 x=0 근방 f<=0=f(0) (극대 0)")
    V.check(1, "4-1 극댓값", 0, V.ans(1, "4-1", "max"))
    # [4](2) 개수: 교점 수를 k 격자로 세어 구간별 개수 확인
    kmin = sp.N(S(V.ans(1, "4-1", "min")), 40)
    u = sp.symbols("u")                                   # x=u^3 (u 는 x 의 실수 세제곱근)
    def nreal(kv):
        rts = sp.Poly(sp.Rational(3, 5)*u**5 - 3*u**2 - kv, u).nroots(n=40, maxsteps=500)
        reals = sorted(float(sp.re(r_)) for r_ in rts if abs(sp.im(r_)) < 1e-8)
        d = []
        for v in reals:
            if not d or abs(v - d[-1]) > 1e-5: d.append(v)
        return len(d)
    c_lt, c_eq_min, c_mid, c_eq0, c_gt0 = nreal(kmin - 1), nreal(kmin), nreal(kmin/2), nreal(0), nreal(1)
    exp_text = f"k<-9*4^(1/3)/5 또는 k>0: {c_lt}개; k=-9*4^(1/3)/5 또는 k=0: {c_eq_min}개; -9*4^(1/3)/5<k<0: {c_mid}개"
    V.report(1, c_lt == c_gt0 == 1 and c_eq_min == c_eq0 == 2 and exp_text == V.ans(1, "4-2"), f"4-2 개수 k<극솟값 {c_lt}, k=극솟값 {c_eq_min}, 사이 {c_mid}, k=0 {c_eq0}, k>0 {c_gt0}")


def p02():
    # [1] b_n 의 a, r 지수를 안쪽부터 쌓아 정확히 계산 (p=2..6, n=1..12) 해 공식과 비교, 극한은 큰 n
    p_, a_, r_, nn = sp.symbols("p a r nn", positive=True)
    bform = sp.sympify(V.ans(2, "1", "b"), locals={"p": p_, "a": a_, "r": r_, "n": nn, "S": sp.Symbol("S")})
    Sform = sp.sympify(V.ans(2, "1", "S"), locals={"p": p_, "n": nn})
    ok = True
    for pv in range(2, 7):
        for N in range(1, 13):
            ea, er = Fraction(0), Fraction(0)             # t_{N+1}=1
            for kk in range(N, 0, -1):
                ea, er = (ea + 1) / pv, (er + (kk - 1)) / pv
            f_a = sp.Rational(1, pv - 1) * (1 - sp.Rational(1, pv)**N)
            f_r = Sform.subs({p_: pv, nn: N})
            ok &= (sp.Rational(ea.numerator, ea.denominator) == f_a) and (sp.simplify(f_r - sp.Rational(er.numerator, er.denominator)) == 0)
    V.report(2, ok, "1 지수(a, r) 정확 비교 p=2..6, n=1..12")
    lim_exp = []
    for pv in range(2, 7):
        N = 400
        ea, er = Fraction(0), Fraction(0)
        for kk in range(N, 0, -1): ea, er = (ea + 1) / pv, (er + (kk - 1)) / pv
        lim_exp.append((float(ea), float(er), 1/(pv - 1), 1/(pv - 1)**2))
    V.report(2, all(abs(u - v) < 1e-9 and abs(w - z) < 1e-9 for u, w, v, z in [(e[0], e[1], e[2], e[3]) for e in lim_exp]), "1 n=400 지수 → 1/(p-1), 1/(p-1)^2")
    limf = sp.sympify(V.ans(2, "1", "limit"), locals={"p": p_, "a": a_, "r": r_})
    V.check(2, "1 극한식(p=3) vs 지수 극한", limf.subs({p_: 3, a_: 5, r_: 2}), 5**sp.Rational(1, 2) * 2**sp.Rational(1, 4))
    # [2] 조건 전수: a 정수, 역함수 존재(f'>=0 모든 x), f(x)=x 가 음이 아닌 서로 다른 세 실근
    ok_a = []
    for av in range(-30, 31):
        f = x**3 + av*x**2 + 3*x
        disc = 4*av**2 - 36
        if disc > 0: continue                                  # f' >= 0 이어야 역함수 존재
        rts = sp.Poly(f - x, x).nroots()
        reals = {round(float(sp.re(r_)), 9) for r_ in rts if abs(sp.im(r_)) < 1e-12}
        if len(reals) == 3 and all(v >= -1e-12 for v in reals): ok_a.append(av)
    V.report(2, ok_a == [-3], f"2 조건 만족 정수 a = {ok_a}")
    # 넓이: f(x)=(x-1)^3+1 이므로 f^{-1}(x)=1+cbrt(x-1) (실수). mpmath 로 수치적분
    import mpmath as mp
    mp.mp.dps = 30
    cbrt = lambda v: mp.sign(v) * abs(v) ** (mp.mpf(1) / 3)
    fa = lambda v: (v - 1)**3 + 1
    fi = lambda v: 1 + cbrt(v - 1)
    area = mp.quad(lambda v: abs(fa(v) - fi(v)), [0, 1, 2])
    V.check(2, "2 넓이", sp.Float(str(area), 25), V.ans(2, "2"), 1e-12)
    V.check(2, "2 f(x) 전개 일치", sp.expand((x - 1)**3 + 1 - (x**3 - 3*x**2 + 3*x)), 0)
    # [3] K 의 범위를 격자로 찾고, 표값으로 확률
    def two_roots_gt1(K):
        D = 64 - 4*(K + 5)
        if D <= 0: return False
        s = math.sqrt(D)
        return (8 - s)/2 > 1
    Ks = [i/1000 for i in range(-5000, 20000)]
    good = [K for K in Ks if two_roots_gt1(K)]
    V.report(2, abs(min(good) - 2) < 2e-3 and abs(max(good) - 11) < 2e-3 or (abs(min(good) - 2) < 2e-3 and max(good) > 10.99), f"3 K 범위 [{min(good)}, {max(good)}] ≈ (2,11)")
    V.check(2, "3 표값", (sp.Rational(6826, 10000) + sp.Rational(9544, 10000)) / 2, V.ans(2, "3"))
    # N(5,9): P(2<K<11)=Φ(2)-Φ(-1)
    phi = lambda z: 0.5*(1 + math.erf(z/math.sqrt(2)))
    V.check(2, "3 정확 확률", phi(2) - phi(-1), sp.N(S(V.ans(2, "3"))), 2e-4)
    # [4](1) 좌우 미분계수
    h = sp.symbols("h", positive=True)
    f_left = lambda hh: -(1 + hh)*sp.log(1 + hh)           # x<1 : g=-x
    f_right = lambda hh: sp.exp(1 + hh)*sp.log(1 + hh)     # x>1 : g=e^x
    dl = sp.limit(f_left(-h)/(-h), h, 0, "+"); dr = sp.limit(f_right(h)/h, h, 0, "+")
    cont = (sp.limit(f_left(-h), h, 0, "+") == 0 == sp.limit(f_right(h), h, 0, "+") == sp.log(1)*sp.pi)
    got = f"continuous={'yes' if cont else 'no'}; differentiable={'yes' if (cont and sp.simplify(dl - dr) == 0) else 'no'}; left={dl}; right={str(dr).replace('E', 'e')}"
    V.report(2, got == V.ans(2, "4-1"), f"4-1 {got} / YAML {V.ans(2, '4-1')}")
    # [4](2) 정적분 정밀 수치
    val = mp.quad(lambda v: (1 + v) * mp.e**v * mp.log(v), [1, mp.e])
    V.check(2, "4-2 정적분", sp.Float(str(val), 25), sp.N(S(V.ans(2, "4-2")), 25), 1e-12)


if __name__ == "__main__": V.run(globals())
