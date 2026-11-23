import sys, pathlib, itertools, math, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy import sqrt, Rational, symbols, S as SS
V = Verifier("UOS-G23B", "items/UOS/PAST")
a_, b_ = sp.symbols("a b", real=True)


def p01():
    # 조건(YAML) 파싱
    def parse(c):
        if "!=" in c:
            l, r = c.split("!=")
            return sp.Gt(sp.Abs(sp.sympify(l, locals={"a": a_, "b": b_}) - sp.sympify(r, locals={"a": a_, "b": b_})), sp.Float("1e-9"))
        return sp.sympify(c, locals={"a": a_, "b": b_})
    conds = [parse(c) for c in V.ans(1, "1-1").split(",")]
    def count_points(a, b):
        m = lambda p: 3*p**2 + 2*a*p + b
        if all(m(float(r)) > 0 for r in [0]) is False: pass
        # f'(x)>0 for all x?  (판별식)
        if 4*a*a - 12*b >= 0:
            return None                      # f'(x)=0 가 되는 x 존재 → 조건 위배
        # PQ = 2|p|>0  <=>  p!=0 and f'(p)^2 = 1/3
        c = b - 1/math.sqrt(3); D = 4*a*a - 12*c
        roots = []
        if D > 1e-12: roots = [(-2*a + s*math.sqrt(D))/6 for s in (1, -1)]
        elif abs(D) <= 1e-12: roots = [-2*a/6]
        roots = [r for r in roots if abs(r) > 1e-9]
        # 직접 거리 확인
        for p in roots:
            mm = m(p); q = p**3 + a*p*p + b*p + 1
            Q = (0, q + p/mm); assert abs(math.hypot(p, Q[1]-q) - 2*abs(p)) < 1e-7
        return len(roots)
    random.seed(1); bad = []; hit = 0
    pts = [(random.uniform(-2, 2), random.uniform(-2, 3)) for _ in range(20000)]
    pts += [(0.0, 1/math.sqrt(3)), (0.5, 1/math.sqrt(3)), (-1.0, 1/math.sqrt(3)), (0.3, 0.3)]
    for a, b in pts:
        n = count_points(a, b)
        ok_formula = all(bool(c.subs({a_: a, b_: b})) for c in conds) if n is not None or True else None
        # 조건식은 a,b에 대해 '점 P가 정확히 2개이고 f'>0'
        truth = (n == 2)
        if truth != ok_formula:
            bad.append((a, b, n, ok_formula))
        hit += truth
    V.report(1, not bad and hit > 100, f"1-1 (a,b) 2만 점 격자: 직접 센 점 P 개수=2 ⇔ 조건, 반례 {bad[:1] or '없음'}, 참 {hit}개")


def tri_area(t):
    s3 = math.sqrt(3)
    def pos(s):
        s %= 3
        if s < 1: return (s, 0.0)
        if s < 2: return (1 - (s-1)/2, (s-1)*s3/2)
        return (0.5 - (s-2)/2, s3/2 - (s-2)*s3/2)
    P, Q, R = pos(t), pos(math.sqrt(2)*t), pos(2*t)
    return abs((Q[0]-P[0])*(R[1]-P[1]) - (R[0]-P[0])*(Q[1]-P[1]))/2


def p02():
    ts = [math.sqrt(2) + i*(2-math.sqrt(2))/200000 for i in range(200001)]
    A = [tri_area(t) for t in ts]
    tmax, tmin = ts[A.index(max(A))], ts[A.index(min(A))]
    V.check(2, "2-1 max 시각", sp.Float(tmax, 6), V.ans(2, "2-1", "max"), tol=1e-4)
    V.check(2, "2-1 min 시각", sp.Float(tmin, 6), V.ans(2, "2-1", "min"), tol=1e-4)
    # 해설지 구간식과 직접 계산 비교
    s3 = math.sqrt(3)
    e1 = max(abs(tri_area(t) - s3*(2-math.sqrt(2))/4*(-t*t+2*t)) for t in ts if t <= 1.5)
    e2 = max(abs(tri_area(t) - s3/4*((2+3*math.sqrt(2))*t*t - (14+5*math.sqrt(2))*t + 18)) for t in ts if t >= 1.5)
    V.report(2, e1 < 1e-9 and e2 < 1e-9, f"2-1 구간식 S(t) 일치 오차 {e1:.1e}, {e2:.1e}")
    t0 = (14+5*sqrt(2))/(2*(2+3*sqrt(2)))
    V.check(2, "t0 정리", t0, (1+16*sqrt(2))/14)
    V.report(2, tri_area(2) > tri_area(math.sqrt(2)), "S(2)>S(sqrt2)")


def p03():
    cells = range(12)
    rows = [[3*i, 3*i+1, 3*i+2] for i in range(4)]
    cols = [[j, j+3, j+6, j+9] for j in range(3)]
    tot = e1 = e2 = c1 = both = 0
    for W in itertools.combinations(cells, 5):
        w = set(W)
        mono = lambda L: len({c in w for c in L}) == 1
        r = [mono(L) for L in rows]; c = [mono(L) for L in cols]
        if any(r) or any(c): tot += 1
        br = sum(1 for L in rows if all(x not in w for x in L)); bc = sum(1 for L in cols if all(x not in w for x in L))
        e1 += br == 1; e2 += br == 2; c1 += bc == 1; both += (br == 1 and bc == 1)
    V.check(3, "3-1 전수", tot, V.ans(3, "3-1"))
    V.report(3, (e1, e2, c1, both) == (432, 36, 168, 72) and 432+36+168-72 == tot,
             f"분해 검산 가로1={e1} 가로2={e2} 세로1={c1} 동시={both}")


def p04():
    from fractions import Fraction as F
    import mpmath as mp
    L = U = F(0); bad = []
    for k in range(1, 401):
        lo = F(1, k+1) + F(1, 2*(k+1)**2); hi = F(1, 2)*(F(1, k)+F(1, k+1))
        mf = lambda q: mp.mpf(q.numerator)/q.denominator
        lg = mp.log(k+1) - mp.log(k)
        if not (mf(lo) <= lg <= mf(hi)): bad.append(("k", k))
        L += lo; U += hi
        if not (mf(L) <= mp.log(k+1) <= mf(U)): bad.append(("n", k))
    X = sp.symbols("x", positive=True); K = sp.symbols("k", positive=True)
    f = -(X-(K+1))/(K+1)**2 + 1/(K+1); g = -(X-K)/(K*(K+1)) + 1/K
    ok = sp.simplify(sp.integrate(f, (X, K, K+1)) - (1/(K+1) + 1/(2*(K+1)**2))) == 0 and \
         sp.simplify(sp.integrate(g, (X, K, K+1)) - (1/K + 1/(K+1))/2) == 0 and \
         sp.simplify(1/X - f - (X-K-1)**2/(X*(K+1)**2)) == 0 and sp.simplify(g - 1/X + (X-K)*(X-K-1)/(X*K*(K+1))) == 0
    V.proof(4, "n=1..400 및 각 k 구간 부등식", bad)
    V.report(4, ok, "접선·현 적분값 및 항등식 sympy 확인")


if __name__ == "__main__": V.run(globals())
