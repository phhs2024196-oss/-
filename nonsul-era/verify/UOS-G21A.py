import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from fractions import Fraction as F
import mpmath as mp
V = Verifier("UOS-G21A", "items/UOS/PAST")
X = sp.Symbol("x"); N = sp.Symbol("n")


def p01():
    # 앞면 수 h 의 마르코프 사슬을 정확한 분수로: 시작 h=2, 모두 같은 면(h=0 또는 5)이면 정지
    # 경로 추적: 첫 선택이 앞면인 경로 표시
    states = {(2, None): F(1)}              # (h, 첫 선택) -> 확률
    PA = PAB = F(0)
    for step in range(1, 5):
        new = {}
        for (h, first), p in states.items():
            for pick_h, q in ((True, F(h, 5)), (False, F(5 - h, 5))):
                if q == 0: continue
                nh = h - 1 if pick_h else h + 1
                f = first if first is not None else pick_h
                if nh in (0, 5):
                    PA += p * q
                    if f: PAB += p * q
                else:
                    new[(nh, f)] = new.get((nh, f), 0) + p * q
        states = new
    V.check(1, "1-1 P(B|A)", sp.Rational(PAB.numerator * PA.denominator, PAB.denominator * PA.numerator), V.ans(1, "1-1"))
    V.report(1, (PA, PAB) == (F(114, 625), F(66, 625)), f"P(A)={PA}, P(A∩B)={PAB}")


def p02():
    mp.mp.dps = 30
    bad = []
    for i in range(1, 100):
        a = mp.mpf(i) / 100
        val = mp.quad(lambda x: (mp.cos(a) - mp.cos(1)*mp.cos(x)) / mp.sin(x)**2, [1 - a, 1 + a])
        # f(x)=x (YAML)
        f = sp.lambdify(X, sp.sympify(V.ans(2, "2-1"), locals={"x": X}), "mpmath")
        if abs(val - 2*mp.sin(f(a))) > 1e-20: bad.append(a)
    V.proof(2, "0<a<1 에서 적분 = 2 sin f(a), f=YAML", bad)


def geom_f(x):
    """원/직사각형 배치를 기하로 직접 풀어 넓이 f(x) 계산(해설의 y,z 공식은 쓰지 않음)."""
    d = 3 + 2*math.sqrt(2)
    z = (d - x) / 2                      # C2 의 왼쪽 접선이 BF 의 연장선 x → 중심 x좌표 d-z = x+z
    # C1 중심 (y, d-y): C2 중심 (d-z, z) 와 외접: dist = y+z → y 이분법
    def g(y): return math.hypot(d - z - y, z - (d - y)) - (y + z)
    lo, hi = 1e-9, d / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(lo) * g(mid) <= 0: hi = mid
        else: lo = mid
    y = (lo + hi) / 2
    # 직사각형 [0,x]x[0,h] 가 C1 의 내부와 겹치지 않는 최대 h: 원 아래쪽 경계의 [0,x] 위 최솟값
    best = float("inf")
    for i in range(0, 20001):
        u = x * i / 20000
        if abs(u - y) <= y:
            best = min(best, (d - y) - math.sqrt(max(y*y - (u - y)**2, 0)))
    return x * best, best, y


def p03():
    d = 3 + 2*math.sqrt(2)
    fpiece = lambda x: x*(d - (1+x)/2 - math.sqrt(x)) if x < 1 else x*(d - 1 - x)
    err = 0; viol = []
    for i in range(1, 400):
        x = (1 + math.sqrt(2)) * i / 400
        f, h, y = geom_f(x)
        err = max(err, abs(f - fpiece(x)))
        if (x < h) != (x < 1 + math.sqrt(2)): viol.append(x)
    V.report(3, err < 1e-4 and not viol, f"3-1 기하 직접 계산과 구간식 최대오차 {err:.1e}, 조건(5) 위반 {viol[:1] or '없음'}")
    # 3-1 YAML 식 자체
    fy = sp.sympify(V.ans(3, "3-1"), locals={"x": X, "Piecewise": sp.Piecewise})
    V.report(3, abs(float(fy.subs(X, 0.37)) - fpiece(0.37)) < 1e-12 and abs(float(fy.subs(X, 2.0)) - fpiece(2.0)) < 1e-12, "YAML 식 = 구간식")
    # 3-2 x=1 좌우 미분계수
    h = sp.Symbol("h")
    fl = X*(3 + 2*sp.sqrt(2) - (1+X)/2 - sp.sqrt(X)); fr = X*(2 + 2*sp.sqrt(2) - X)
    V.check(3, "3-2 좌미분계수", sp.simplify(sp.diff(fl, X).subs(X, 1)), 2*sp.sqrt(2))
    V.check(3, "3-2 우미분계수", sp.simplify(sp.diff(fr, X).subs(X, 1)), 2*sp.sqrt(2))
    V.check(3, "3-2 연속", sp.simplify((fl - fr).subs(X, 1)), 0)


def p04():
    def cnt(n):
        c = 0
        for a in range(1, 6*n + 1):
            for b in range(a + 1, 6*n + 1):
                cc = 2*b - a
                if b < cc <= 6*n:
                    s = lambda k: math.sin(math.pi*k/(6*n))
                    if s(b) - s(a) > 1e-12 and s(cc) - s(b) > 1e-12: c += 1
        return c
    f = sp.lambdify(N, sp.sympify(V.ans(4, "4-1"), locals={"n": N}), "math")
    bad = [(n, cnt(n), f(n)) for n in range(1, 31) if cnt(n) != f(n)]
    V.report(4, not bad, f"4-1 n=1..30 전수 {bad[:1] or '일치'}")
    # 해설 단계: (i)+(ii)
    n = sp.Symbol("n", positive=True); b = sp.Symbol("b", integer=True)
    tot = sp.summation(b - 1, (b, 2, 2*n)) + sp.summation(6*n - 2*b - 1, (b, 2*n + 1, 3*n - 1))
    V.check(4, "구간합", sp.expand(tot), sp.expand(sp.sympify(V.ans(4, "4-1"), locals={"n": n})))


if __name__ == "__main__": V.run(globals())
