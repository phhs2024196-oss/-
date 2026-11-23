#!/usr/bin/env python3
"""UOS-M20 (2020학년도 모의논술 자연계열 4문항) — sympy/수치 검산. 기대값은 YAML 정답(V.ans)에서 읽는다."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np

V = Verifier("UOS-M20", "items/UOS/PAST")
a_pos = sp.Symbol("a", positive=True)
LOC = {"a": a_pos}


def p01():
    # 극한식: a=1,2,3 에서 S_n 을 n 의 다항식으로 정확히 구해 극한 계산, a=1/2 는 큰 n 수치
    expr = sp.sympify(V.ans(1, "1-1"), locals=LOC)
    k_, l_, m_ = sp.symbols("k l m", integer=True, positive=True)
    nn = sp.Symbol("n", positive=True, integer=True)
    for av in (1, 2, 3):
        # |k-l| 를 구간으로 분해: l<k, l>=k
        ak = sp.summation((k_ - l_) ** av, (l_, 1, k_ - 1)) + sp.summation((l_ - k_) ** av, (l_, k_, 2 * nn))
        Sn = sp.summation(sp.expand(sp.simplify(ak)), (k_, 1, nn))
        lim = sp.limit(Sn / nn ** (av + 2), nn, sp.oo)
        V.check(1, f"a={av} 극한", lim, expr.subs(a_pos, av))
    # 수치(직접 이중합)
    for av, N in ((0.5, 3000), (1.7, 3000)):
        ks = np.arange(1, N + 1)[:, None]
        ls = np.arange(1, 2 * N + 1)[None, :]
        Sn = (np.abs(ks - ls) ** av).sum()
        val = Sn / N ** (av + 2)
        ex = float(expr.subs(a_pos, av))
        V.report(1, abs(val - ex) / ex < 2e-3, f"a={av} n={N} 수치 {val:.6f} / 식 {ex:.6f}")


def p02():
    ans = int(V.ans(2, "2-1"))
    f = lambda t: 4 * np.cos(4 * t) - 8 * np.cos(2 * t)
    T = np.linspace(0, 2 * np.pi, 2_000_001)
    # 넓이 (1/2)|sin(angleP-angleQ)| 최대 <=> |sin f|=1 <=> cos f = 0
    c = np.cos(f(T))
    cnt = int(np.sum(c[:-1] * c[1:] < 0))
    zeros = np.flatnonzero(c == 0)
    # 접하는(겹근) 경우 없음 확인: f 의 극값이 (2n+1)pi/2 와 다름
    ext = [-4, -6, 12]
    ok_ext = all(abs(((e / (math.pi / 2)) - 1) / 2 - round(((e / (math.pi / 2)) - 1) / 2)) > 1e-6 for e in ext)
    # 직접: 삼각형 넓이 = (1/2)|sin(4cos4t-8cos2t)| 의 최댓값이 1/2 인 점의 개수 (넓이 재계산)
    P = np.stack([np.cos(4 * np.cos(4 * T)), np.sin(4 * np.cos(4 * T))])
    Q = np.stack([np.cos(8 * np.cos(2 * T)), np.sin(8 * np.cos(2 * T))])
    area = 0.5 * np.abs(P[0] * Q[1] - P[1] * Q[0])
    d = area[1:-1] - area[:-2], area[2:] - area[1:-1]
    peaks = int(np.sum((d[0] > 0) & (d[1] <= 0) & (area[1:-1] > 0.5 - 1e-6)))
    V.check(2, "넓이=1/2 인 t 의 개수(cos f 부호변화)", cnt + len(zeros), ans)
    V.check(2, "넓이 최대(=0.5) 극대점 개수(직접 넓이)", peaks, ans)
    V.report(2, ok_ext, "f 의 극값 -4,-6,12 는 (2n+1)π/2 와 달라 접점 없음")
    # 치환 풀이: x=cos2t, h(x)=8x^2-8x-4, 정수 n 별 근 개수
    X = sp.Symbol("X", real=True)
    h = 8 * X**2 - 8 * X - 4
    total = 0
    for nv in range(-6, 8):
        v = sp.Rational(2 * nv + 1, 2) * sp.pi
        sols = [r for r in sp.solve(sp.Eq(h, v), X) if r.is_real and -1 <= float(r) <= 1]
        for r in sols:
            assert abs(float(r)) < 1
        total += len(sols)
    V.check(2, "치환 풀이: 근 7개 × 4", total * 4, ans)


def _gd(a, xx):
    f = xx**3 - a**2 * xx
    return min(abs(f), abs(xx))


def _kinks(a):
    """g=min(|f|,|x|) 의 꺾임점: 후보(영점, |f|=|x| 의 교점) 중 좌우 미분계수가 다른 점."""
    xs = sp.Symbol("xs", real=True)
    f = xs**3 - a**2 * xs
    cand = {sp.Integer(0)} | set(sp.solve(f, xs)) | set(sp.solve(f - xs, xs)) | set(sp.solve(f + xs, xs))
    cand = {c for c in cand if c.is_real}
    out = []
    h = 1e-7
    for c in cand:
        c = float(c)
        dl = (_gd(a, c) - _gd(a, c - h)) / h
        dr = (_gd(a, c + h) - _gd(a, c)) / h
        if abs(dl - dr) > 1e-3:
            out.append(c)
    return sorted(set(round(v, 6) for v in out))


def _parse_expected(a):
    """YAML 정답 문자열에서 a 값에 해당하는 미분불가점 목록."""
    s = V.ans(3, "3-1")
    for part in s.split(";"):
        cond, val = (p.strip() for p in part.split("->"))
        if cond == "a=0":
            hit = a == 0
        elif cond == "0<a<=1":
            hit = 0 < a <= 1
        elif cond == "a>1":
            hit = a > 1
        else:
            raise ValueError(cond)
        if hit:
            pts = []
            for tok in val.replace("x=", "").split(","):
                tok = tok.strip()
                e = sp.sympify(tok[2:] if tok.startswith("+-") else tok, locals={"a": sp.Rational(a).limit_denominator(1000)})
                pts += [float(e), -float(e)] if tok.startswith("+-") else [float(e)]
            return sorted(set(round(v, 6) for v in pts))
    raise ValueError("해당 구간 없음")


def p03():
    for a in (0, sp.Rational(1, 2), 1, sp.Rational(3, 2), 2, 3):
        got = _kinks(a)
        exp = _parse_expected(float(a))
        V.report(3, got == exp, f"3-1 a={a}: 꺾임점 {got} / YAML {exp}")
    # 3-2: 둘러싸인 영역의 넓이 = ∫|g-f| (구간 [-a,a] 에서 f,g 가 만나는 두 점 사이)
    import mpmath as mp
    expr = sp.sympify(V.ans(3, "3-2"), locals=LOC)
    for av in (0.5, 1, 1.5, 2, 3):
        f = lambda x: x**3 - av**2 * x
        g = lambda x: min(abs(f(x)), abs(x))
        pts = sorted({-av, 0.0, av, math.sqrt(max(av**2 - 1, 0)), -math.sqrt(max(av**2 - 1, 0))})
        pts = [p for p in pts if -av <= p <= av]
        area = mp.quad(lambda x: abs(g(x) - f(x)), pts)
        V.report(3, abs(float(area) - float(expr.subs(a_pos, av))) < 1e-8,
                 f"3-2 a={av}: 수치 넓이 {float(area):.8f} / 식 {float(expr.subs(a_pos, av)):.8f}")
        # 구간 밖에서는 f=g 인 부분 외에 닫힌 영역이 생기지 않는다: x>a 에서 g-f 부호 확인(같거나 f가 위)
        r = math.sqrt(av**2 + 1)
        xs_ = np.linspace(av, r, 50)
        V.report(3, all(abs(g(v) - f(v)) < 1e-9 for v in xs_), f"a={av}: [a,√(a²+1)] 에서 f=g")


def p04():
    ans = sp.sympify(V.ans(4, "4-1"))
    # (1) 해설 매개화 PQ*QR = 3a(1-a)/(sqrt3+2 sin2θ) 를 sympy 로 최대화
    A, th = sp.symbols("A th", real=True)
    prod = 3 * A * (1 - A) / (sp.sqrt(3) + 2 * sp.sin(2 * th))
    best = max(float(prod.subs({A: sp.Rational(1, 2), th: tv})) for tv in (sp.pi / 6, sp.pi / 3))
    V.check(4, "해설 식 a=1/2,θ=π/6 의 값", prod.subs({A: sp.Rational(1, 2), th: sp.pi / 6}), ans)
    # (2) 독립 수치: 정삼각형 안의 (회전한) 직사각형 최대 넓이 — 무작위 탐색 + 국소 개선
    rng = np.random.default_rng(1)
    S3 = math.sqrt(3)
    tri = np.array([[0, 0], [1, 0], [0.5, S3 / 2]])

    def inside(pts, eps=1e-12):
        # 변 AB: y>=0, BC: sqrt3(1-x) - y >= 0, CA: y - ... : sqrt3 x - y >= 0
        x_, y_ = pts[..., 0], pts[..., 1]
        return (y_ >= -eps) & (S3 * (1 - x_) - y_ >= -eps) & (S3 * x_ - y_ >= -eps)

    # 반평면 n.p <= b (바깥 법선): AB: (0,-1)<=0, BC: (sqrt3,1)<=sqrt3, CA: (-sqrt3,1)<=0
    N = np.array([[0, -1.0], [S3, 1.0], [-S3, 1.0]])
    Bv = np.array([0.0, S3, 0.0])
    CX, CY, W = np.meshgrid(np.linspace(0.1, 0.9, 161), np.linspace(0.02, 0.8, 157), np.linspace(0.01, 0.6, 120), indexing="ij")

    def max_area(phi):
        u, v = np.array([math.cos(phi), math.sin(phi)]), np.array([-math.sin(phi), math.cos(phi)])
        hmax = np.full(CX.shape, np.inf)
        for nv, bv in zip(N, Bv):
            nu, nvv = abs(nv @ u), abs(nv @ v)
            slack = bv - (nv[0] * CX + nv[1] * CY) - W * nu
            if nvv < 1e-12:
                hmax = np.where(slack < 0, -1.0, hmax)
            else:
                hmax = np.minimum(hmax, slack / nvv)
        hmax = np.where(hmax < 0, 0.0, hmax)
        return float(np.max(4 * W * hmax))

    phis = [i * math.pi / 180 for i in range(0, 91, 3)]
    ma = {round(p_, 4): max_area(p_) for p_ in phis}
    gbest = max(ma.values())
    V.report(4, abs(gbest - float(ans)) < 3e-3 and gbest <= float(ans) + 1e-6,
             f"방향별 격자 탐색 최대 {gbest:.5f} (정답 {float(ans):.5f}) 방향 {max(ma, key=ma.get):.3f} 에서 최대")
    # (3) 해석적 기준: 정삼각형 내접 직사각형 최대 넓이 = 삼각형 넓이의 1/2
    V.check(4, "삼각형 넓이/2", sp.sqrt(3) / 4 / 2, ans)


if __name__ == "__main__":
    V.run(globals())
