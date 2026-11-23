import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DDW-M26", "items/DDW/PAST")


def p01():
    """내접 사각형: cosA=1/4, BD=2√3, [BCD]=5/2[OBD]. R, cos∠BOD, 넓이, 세 변의 곱을 독립 재계산."""
    cosA = sp.Rational(1, 4)
    sinA = sp.sqrt(1 - cosA**2)
    BD = 2*sp.sqrt(3)
    R = sp.simplify(BD/(2*sinA))
    V.check(1, "R", R, 4*sp.sqrt(5)/5)
    # 중심각: 좌표로 직접 계산 — O=(0,0), B,D 는 현 BD 가 길이 2√3 인 원 위의 점
    half = BD/2
    h0 = sp.sqrt(R**2 - half**2)            # 중심에서 현까지의 거리
    Bp, Dp = sp.Matrix([-half, -h0]), sp.Matrix([half, -h0])
    cosBOD = sp.simplify(Bp.dot(Dp)/R**2)
    V.check(1, "cos∠BOD=cos2A", cosBOD, 2*cosA**2 - 1)
    S1 = sp.simplify(sp.Rational(1, 2)*BD*h0)           # △OBD 밑변 BD·높이
    S2 = sp.Rational(5, 2)*S1
    V.check(1, "△BCD 넓이", S2, sp.sqrt(15)/2)
    # BC, CD 를 직접 구해 존재 확인(C 는 A 의 맞은편 호: cosC=-cosA)
    cosC, sinC = -cosA, sinA
    pr = sp.simplify(2*S2/sinC)                          # BC*CD
    ssum = 12 + 2*pr*cosC                                # BC^2+CD^2 = BD^2 + 2 BC CD cos C
    u = sp.symbols("u", positive=True)
    rts = sp.solve(sp.Eq(u**2 - ssum*u + pr**2, 0), u)   # u = BC^2 , CD^2
    V.check(1, "BC^2,CD^2 양의 실수", all(r.is_real and r > 0 for r in rts), True)
    BC, CD = [sp.sqrt(r) for r in rts]
    V.check(1, "BC*CD", sp.simplify(BC*CD), pr)
    prod = sp.simplify(BC*CD*BD)
    V.check(1, "세 변의 곱", prod, V.ans(1, "1"))
    # abc/(4R) = 넓이 와 삼각형 BCD 의 외접원이 원 O 와 일치
    V.check(1, "넓이=abc/4R", sp.simplify(prod/(4*R)), S2)
    V.check(1, "△BCD 사인법칙 BD/sinC=2R", sp.simplify(BD/sinC), 2*R)


def p02():
    nn = sp.symbols("nn", positive=True, integer=True)
    kk = sp.symbols("kk", positive=True, integer=True)
    ans = sp.sympify(V.ans(2, "2-1"), locals={"x": x, "f": sp.Function("f")})
    # (1): 여러 연속함수 f 에 대해 급수의 극한 = ∫_0^1 x^2 f(x) dx
    for fx in (x, x**2 + 1, sp.exp(x)):
        Sn = sp.summation(kk**2*fx.subs(x, kk/nn), (kk, 1, nn))/nn**3
        try:
            lim = sp.limit(sp.simplify(Sn), nn, sp.oo)
        except Exception:
            lim = None
        target = ans.replace(sp.Function("f"), sp.Lambda(x, fx)).doit()
        if lim is None:
            import mpmath as mp
            N = 20000
            num = sum(mp.mpf(j)**2*sp.N(fx.subs(x, sp.Rational(j, N))) for j in range(1, N + 1))/mp.mpf(N)**3
            V.check(2, f"2-1 f={fx} 수치", abs(num - sp.N(target)) < 1e-3, True)
        else:
            V.check(2, f"2-1 f={fx}", lim, target)
    # (2)
    val = sp.integrate(x**2*sp.sin(sp.pi*x), (x, 0, 1))
    V.check(2, "2-2 적분", val, V.ans(2, "2-2"))
    import mpmath as mp
    N = 200000
    num = sum(mp.mpf(j)**2*mp.sin(mp.pi*mp.mpf(j)/N) for j in range(1, N + 1))/mp.mpf(N)**3
    V.check(2, "2-2 리만합 수치", abs(num - sp.N(val)) < 1e-4, True)
    F = -x**2/sp.pi*sp.cos(sp.pi*x) + 2*x/sp.pi**2*sp.sin(sp.pi*x) + 2/sp.pi**3*sp.cos(sp.pi*x)
    V.check(2, "2-2 부정적분 미분", sp.simplify(sp.diff(F, x) - x**2*sp.sin(sp.pi*x)), 0)


def p03():
    xs = x
    f = lambda z: sp.Abs(z**2 - z - 12)
    # 연속 조건을 만족하는 정수 k 전수 조사
    ks = [kv for kv in range(-60, 61) if 2*f(sp.Integer(kv)) == f(sp.Integer(0))]
    V.check(3, "연속 만족 정수 k", ks, [-2, 3])

    def kinks(kv):
        """g 의 미분불가능한 점(좌우 미분계수가 다른 점)을 구한다."""
        q_left = (xs + kv)**2 - (xs + kv) - 12          # x<0 조각 2|q_left|
        q_right = xs**2 - xs - 12                         # x>=0 조각 |q_right|
        pts = []
        for r in sp.solve(q_left, xs):
            if r < 0:
                pts.append(r)
        for r in sp.solve(q_right, xs):
            if r > 0:
                pts.append(r)
        # x=0: 좌우 미분계수
        dl = sp.diff(2*sp.Abs(q_left), xs).subs(xs, 0) if q_left.subs(xs, 0) != 0 else None
        dl = 2*sp.sign(q_left.subs(xs, 0))*sp.diff(q_left, xs).subs(xs, 0)
        dr = sp.sign(q_right.subs(xs, 0))*sp.diff(q_right, xs).subs(xs, 0)
        if dl != dr:
            pts.append(sp.Integer(0))
        return sorted(pts)
    good = []
    for kv in ks:
        P = kinks(kv)
        V.check(3, f"k={kv} 꺾임점 3개", len(P), 3)
        h = sp.expand(sp.prod([xs - p for p in P]))
        ok = sp.diff(h, xs).subs(xs, 4) == 20
        # 곱 gh 가 실제로 꺾임점에서 미분가능한지 직접 확인(좌우 미분계수)
        for p in P:
            if p < 0:
                gl = 2*sp.Abs((xs + kv)**2 - (xs + kv) - 12)
                gr = gl if p < 0 else None
            # 한쪽 조각의 식으로 좌우 도함수 비교
            if p != 0:
                piece = 2*sp.Abs((xs + kv)**2 - (xs + kv) - 12) if p < 0 else sp.Abs(xs**2 - xs - 12)
                d_minus = sp.limit(sp.diff(piece*h, xs), xs, p, "-")
                d_plus = sp.limit(sp.diff(piece*h, xs), xs, p, "+")
                V.check(3, f"k={kv} gh 가 x={p} 에서 미분가능", d_minus, d_plus)
        if ok:
            good.append((kv, h))
        V.check(3, f"k={kv} h'(4)", sp.diff(h, xs).subs(xs, 4), 20 if kv == -2 else 40)
    V.check(3, "h'(4)=20 인 k 는 하나", [kv for kv, _ in good], [-2])
    kv, h = good[0]
    V.check(3, "3-1 h", h, sp.sympify(V.ans(3, "3-1", "h"), locals={"x": xs}))
    V.check(3, "3-1 h(k)", h.subs(xs, kv), V.ans(3, "3-1", "hk"))
    # 3-2
    a_, b_ = sp.symbols("a b")
    sol = sp.solve([sp.Eq(h.subs(xs, 2), 2*a_ + b_), sp.Eq(sp.diff(h, xs).subs(xs, 2), a_)], [a_, b_])
    j = sol[a_]*xs + sol[b_]
    V.check(3, "j", j, -4*xs - 4)
    D = sp.expand(h - j)
    crit = sp.solve(sp.diff(D, xs), xs)
    vals = {c: D.subs(xs, c) for c in crit}
    mx = max(vals.items(), key=lambda kv_: kv_[1])
    mn = min(vals.items(), key=lambda kv_: kv_[1])
    V.check(3, "극댓값", mx[1], V.ans(3, "3-2", "max"))
    V.check(3, "극솟값", mn[1], V.ans(3, "3-2", "min"))
    V.check(3, "극값 위치(0,2)", [mx[0], mn[0]], [0, 2])
    V.check(3, "증감 부호 D'(-1),D'(1),D'(3)", [sp.sign(sp.diff(D, xs).subs(xs, v)) for v in (-1, 1, 3)], [1, -1, 1])

if __name__ == "__main__": V.run(globals())
