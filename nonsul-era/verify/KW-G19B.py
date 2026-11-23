"""KW-G19B 검산(2019학년도 오후 문제 1·2): 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, math, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-G19B", "items/KW/PAST")
mp.mp.dps = 60
LOC = {"x": x, "e": sp.E, "R": sp.Symbol("R"), "n": sp.Symbol("n", positive=True)}


def f1(u):
    return mp.e ** u if u >= 0 else 2 * u + 1


def p01():
    # [1](1) 구간별 합성: 세 구간 표본에서 YAML 식과 비교
    pw = {k: sp.sympify(V.ans(1, "1-1", k), locals=LOC) for k in ("p1", "p2", "p3")}
    for u in (0, 0.3, 1.2, -0.1, -0.5, -0.49, -0.9, -2):
        key = "p1" if u >= 0 else ("p2" if u >= -0.5 else "p3")
        V.check(1, f"1-1 x={u}", float(f1(f1(mp.mpf(u)))), float(pw[key].subs(x, u)), 1e-9)
    # [1](2) y=0,1 사이, x<=0, 곡선 오른쪽: 가로띠 적분 (y 에 대해 x(y) 구함)
    def xofy(y):
        return mp.findroot(lambda z: f1(f1(z)) - y, -0.5 if y > 0.5 else -0.9)
    area = mp.quad(lambda y: 0 - xofy(y), [0, 1])
    V.check(1, "1-2 넓이", sp.nsimplify(float(area), rational=True, tolerance=1e-12), sp.sympify(V.ans(1, "1-2")), 1e-9)
    # [2](1) 격자 검사(경계는 정확값)
    pred = V.ans(1, "2-1").replace("sqrt", "math.sqrt").replace("or", " or ")
    def holds(xx):
        fl = mp.floor
        f = xx * xx - xx - 3; g = xx * xx - xx + 5
        return abs(2 * fl(f) - 5) <= fl(g) + 3
    s5 = math.sqrt(5)
    bad = []
    pts = [i / 997 for i in range(-6 * 997, 7 * 997)] + [-4, 5, (1 - s5) / 2, (1 + s5) / 2]
    for xx in pts:
        want = eval(pred, {"x": xx, "math": math})
        if holds(mp.mpf(xx)) != want:
            bad.append(xx)
    V.check(1, "2-1 격자 불일치 수", len(bad), 0)
    # 경계 정확성: 끝점 x=(1±√5)/2 는 포함, -4·5 는 제외
    V.check(1, "2-1 끝점 포함", [int(holds((1 - mp.sqrt(5)) / 2)), int(holds((1 + mp.sqrt(5)) / 2)), int(holds(mp.mpf(-4))), int(holds(mp.mpf(5)))], [1, 1, 0, 0])
    # [2](2) 좌·우극한 (a±1e-30 에서 원래 정의대로 계산)
    def E(xx):
        f = xx * xx - xx - 3; g = xx * xx - xx + 5
        return 2 * mp.floor(g + xx) / (mp.floor(f + xx) + 5)
    def exists(a):
        d = mp.mpf(10) ** -30
        return E(a - d) == E(a + d)
    rngs = [r.strip() for r in V.ans(1, "2-2").split(";")]
    def inpred(a):
        return any(eval(r.replace("sqrt", "math.sqrt"), {"a": a, "math": math}) for r in rngs)
    bad = []
    tests = [mp.mpf(i) / 997 for i in range(1, int(1.7320508 * 997))] + [mp.mpf(1), mp.sqrt(2)]
    for a in tests:
        if exists(a) != inpred(float(a)) and abs(float(a) - 1) > 1e-9 and abs(float(a) - 2 ** .5) > 1e-9:
            bad.append(float(a))
    V.check(1, "2-2 일반점 불일치", len(bad), 0)
    V.check(1, "2-2 a=1,√2 극한 없음", [int(exists(mp.mpf(1))), int(exists(mp.sqrt(2)))], [0, 0])
    V.check(1, "2-2 a=1,√2 제외", [int(inpred(1.0)), int(inpred(2 ** .5))], [0, 0])
    # [3](1) 나머지: 무작위 다항식
    X = sp.Symbol("x"); random.seed(1)
    rem_f = sp.sympify(V.ans(1, "3-1"), locals=LOC)
    ok = True
    for _ in range(30):
        P = sum(random.randint(-9, 9) * X**i for i in range(5))
        Rv = P.subs(X, sp.Rational(5, 2))
        ok &= sp.simplify(sp.rem(sp.expand((X + 1) * P), 2 * X - 5, X) - rem_f.subs(sp.Symbol("R"), Rv)) == 0
    V.check(1, "3-1 나머지", int(ok), 1)
    # [3](2) [증명-수치확인] 삼차식이 [0,3]의 41개 격자 모두에서 0 인 경우
    cex = []
    for _ in range(3000):
        cs = [random.choice([0, 0, random.randint(-5, 5)]) for _ in range(3)] + [random.choice([1, -1, 2, 3])]
        P = sum(c * X**i for i, c in enumerate(cs))
        if all(P.subs(X, sp.Rational(i, 10)) == 0 for i in range(0, 31)):
            cex.append(cs)
    # 인수정리 구성: x(x-1)(x-2)(x-3)Q 는 차수 4 이상
    for k in range(1, 4):
        Q = sum(random.randint(1, 3) * X**i for i in range(k))
        if sp.degree(sp.expand(X * (X - 1) * (X - 2) * (X - 3) * Q), X) <= 3:
            cex.append(("deg", k))
    V.proof(1, "3-2 참(귀류)", cex)


def p02():
    n = sp.Symbol("n", positive=True)
    # [1](1) Q_n: 기울기 -1 직선과 y=3^x 의 교점 (수치)
    qx = sp.sympify(V.ans(2, "1-1", "Qx"), locals=LOC); qy = sp.sympify(V.ans(2, "1-1", "Qy"), locals=LOC)
    ok = True
    for nv in (1, 2, 3, 5, 10, 27, 100):
        L = mp.log(nv, 3)
        xs = mp.findroot(lambda z: 3 ** z - (L + nv - z), (-5, 8), solver="anderson")
        ok &= abs(xs - float(qx.subs(n, nv))) < 1e-12 and abs(3 ** xs - float(qy.subs(n, nv))) < 1e-9
    V.check(2, "1-1 Q_n", int(ok), 1)
    # [1](2) 신발끈 넓이
    an = sp.sympify(V.ans(2, "1-2"), locals=LOC)
    ok = True
    for nv in (1, 2, 3, 7, 50):
        L = mp.log(nv, 3)
        P = (nv, L); Q = (L, nv)
        area = abs(P[0] * Q[1] - P[1] * Q[0]) / 2
        ok &= abs(area - float(an.subs(n, nv))) < 1e-10
    V.check(2, "1-2 a_n", int(ok), 1)
    # [1](3)
    sf = sp.sympify(V.ans(2, "1-3"), locals=LOC)
    ok = True
    for nv in range(1, 40):
        tot = sum(2 * (k**2 / 2 - mp.log(k, 3) ** 2 / 2) + mp.log(k, 3) ** 2 for k in range(1, nv + 1))
        ok &= abs(tot - float(sf.subs(n, nv))) < 1e-9
    V.check(2, "1-3 합", int(ok), 1)
    # [2](1) A, B 의 교점: 실근 전부
    X, Y = sp.symbols("X Y", real=True)
    sol = sp.solve([X**2 - 4 * Y + 4, Y**2 - 4 * X + 4], [X, Y], dict=True)
    real = [(s[X], s[Y]) for s in sol if s[X].is_real and s[Y].is_real]
    V.proof(2, "2-1 교점 (2,2) 하나뿐", [r for r in real if r != (2, 2)] + ([] if (2, 2) in real else ["(2,2) 없음"]))
    # [2](2) 부피
    S = sp.sqrt(3) / 4 * (x**2 / 4 + 1 - x) ** 2
    V.check(2, "2-2 V", sp.simplify(sp.integrate(S, (x, 0, 1))), sp.sympify(V.ans(2, "2-2")))
    # [3] a, b 범위, 극대·극소
    a = sp.sympify(V.ans(2, "3", "a"))
    V.check(2, "3 a", a, sp.Rational(1, 3))
    pred_s = [t.strip() for t in V.ans(2, "3").split(";") if t.strip().startswith("b")][0]
    bad = []
    for i in range(-600, 600):
        bv = i / 100
        us = [-bv + math.sqrt(bv * bv - 1), -bv - math.sqrt(bv * bv - 1)] if bv * bv >= 1 else []
        exist = any(0 < u < 1 - 1e-12 for u in us)   # cosθ=u (θ≠0) 로 두 근 cosθ, secθ=1/u
        if exist != eval(pred_s, {"b": bv}):
            bad.append(bv)
    V.check(2, "3 b 범위 불일치 수", len(bad), 0)
    th = 0.7; bv = -(math.cos(th) + 1 / math.cos(th)) / 2
    fp = lambda u: 3 * a * u * u + 2 * bv * u + 1
    c, s_ = math.cos(th), 1 / math.cos(th)
    mx = "cos(theta)" if fp(c - .01) > 0 > fp(c + .01) else "min"
    mn = "sec(theta)" if fp(s_ - .01) < 0 < fp(s_ + .01) else "max"
    V.check(2, "3 극대 위치", mx == V.ans(2, "3", "max"), True)
    V.check(2, "3 극소 위치", mn == V.ans(2, "3", "min"), True)
    # [4] sup a = inf tan3x/x = 3
    inf = min(mp.tan(3 * mp.mpf(i) * (mp.pi / 6) / 4000) / (mp.mpf(i) * (mp.pi / 6) / 4000) for i in range(1, 4000))
    V.check(2, "4 하한 근사", float(inf), float(sp.sympify(V.ans(2, "4"))), 1e-4)
    V.check(2, "4 극한", sp.limit(sp.tan(3 * x) / x, x, 0, "+"), sp.sympify(V.ans(2, "4")))
    am = float(sp.sympify(V.ans(2, "4")))
    holds = all(mp.tan(3 * u) - am * u > 0 for u in [mp.mpf(i) * (mp.pi / 6) / 2000 for i in range(1, 2000)])
    fails = any(mp.tan(3 * u) - (am + 0.01) * u <= 0 for u in [mp.mpf(i) * (mp.pi / 6) / 20000 for i in range(1, 400)])
    V.check(2, "4 a=3 성립·a>3 불성립", [int(holds), int(fails)], [1, 1])


if __name__ == "__main__": V.run(globals())
