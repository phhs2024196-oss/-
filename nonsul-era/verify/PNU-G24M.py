#!/usr/bin/env python3
"""PNU-G24M (부산대 2024학년도 지역인재전형(논술) 의·약학계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML answer 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "PNU-G24M"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/PNU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[증명-수치확인] (1-1) 0<x-sin x<x^3/6, (1-2) b_n<13/12 와 공식 풀이의 중간 등식·부등식."""
    import mpmath as mp
    mp.mp.dps = 150
    x = sp.symbols("x", positive=True)
    # (1-1) 기호: 보조함수 미분, 격자 1000점 확인
    f = x - sp.sin(x); g = x**3 / 6 - x + sp.sin(x)
    sym_ok = sp.simplify(sp.diff(g, x, 2) - f) == 0 and f.subs(x, 0) == 0 and g.subs(x, 0) == 0
    grid = [mp.mpf(i) / 1000 for i in range(1, 1000)]
    ok1 = sym_ok and md_answer(1, "1") is None and all(0 < t - mp.sin(t) < t**3 / 6 for t in grid)
    # (1-2) 설정에서 직접: 호 AB(중심각 1) 2^n 등분, 삼각형 OBP_{2^n-1} 넓이 = 1/2*|OB x OP|
    def T(n):
        th = mp.mpf(1); step = th / 2**n
        B = (mp.cos(th), mp.sin(th)); P = (mp.cos(th - step), mp.sin(th - step))
        return abs(B[0] * P[1] - B[1] * P[0]) / 2
    N = 80
    S = {n: 2**n * T(n) for n in range(1, N + 1)}
    a = {1: S[1]}
    a.update({n: S[n] - S[n - 1] for n in range(2, N + 1)})
    b = {n: sum(2**k * a[k] for k in range(1, n + 1)) for n in range(1, N + 1)}
    bound = mp.mpf(13) / 12
    ok_b = all(b[n] < bound for n in b)
    ok_S = all(S[n] < mp.mpf(1) / 2 and mp.mpf(1) / 2 - S[n] < 1 / (12 * mp.mpf(4)**n) for n in S)
    ok_abel = all(abs(b[n] - (2**n * S[n] - sum(2**k * S[k] for k in range(1, n)))) < mp.mpf(10)**-25 for n in b)
    ok_a1 = abs(a[1] - mp.sin(mp.mpf(1) / 2)) < mp.mpf(10)**-50          # 공식 채점표의 '2 sin(1/2)' 오기 확인
    ok2 = ok_b and ok_S and ok_abel and ok_a1 and md_answer(1, "2") is None
    report(1, ok1 and ok2, f"[증명-수치확인] 1-1 grid ok={ok1}; b_1={mp.nstr(b[1], 10)} b_80={mp.nstr(b[N], 12)} < 13/12={mp.nstr(bound, 12)}; S_n<1/2, abel ok={ok_abel}, a1=sin(1/2)")


def p02():
    t = sp.symbols("t", real=True)
    X, Y = sp.symbols("X Y", real=True)
    # (2-1) 법선 방향 단위벡터 중 y 성분이 양인 쪽으로 1 이동
    nvec = sp.Matrix([-t, 1]) / sp.sqrt(t**2 + 1)
    F = t + nvec[0]; G = t**2 / 2 + nvec[1]
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "1").replace(" ", "").split(";"))
    mdF = sp.sympify(parts["f(t)"], locals={"t": t}); mdG = sp.sympify(parts["g(t)"], locals={"t": t})
    ok1 = sp.simplify(F - mdF) == 0 and sp.simplify(G - mdG) == 0
    # 조건 확인: 거리 1, 수직
    ok1 = ok1 and sp.simplify((F - t)**2 + (G - t**2 / 2)**2 - 1) == 0 and sp.simplify((F - t) + t * (G - t**2 / 2)) == 0
    # (2-2) D 의 접선과 포물선 교점 → 길이 → 극한
    slope = sp.simplify(sp.diff(G, t) / sp.diff(F, t))
    quad = sp.expand(X**2 / 2 - (G + slope * (X - F)))
    roots = sp.solve(quad, X)
    l = sp.sqrt(1 + slope**2) * sp.Abs(roots[0] - roots[1])
    tp = sp.symbols("tp", positive=True)
    lim = sp.limit((l / (t * sp.sqrt(t))).subs(t, tp), tp, sp.oo)
    ok2 = sp.simplify(lim - sp.sympify(md_answer(2, "2"))) == 0
    # (2-3) 직접 수치적분(치환 없이)
    import mpmath as mp
    mp.mp.dps = 30
    integrand = sp.lambdify(t, F * sp.diff(G, t) / t, "mpmath")
    val = mp.quad(integrand, [mp.sqrt(3), 2 * mp.sqrt(2)])
    ok3 = abs(val - mp.mpf(sp.Rational(md_answer(2, "3")).p) / sp.Rational(md_answer(2, "3")).q) < mp.mpf(10)**-20
    report(2, ok1 and ok2 and ok3, f"slope={slope} lim={lim} integral={mp.nstr(val, 15)}")


def p03():
    import mpmath as mp
    mp.mp.dps = 30
    s3 = sp.sqrt(3)
    A = sp.Matrix([0, 0, 0]); B = sp.Matrix([4, 0, 0]); C = sp.Matrix([2, 2 * s3, 0])
    up = sp.Matrix([0, 0, 8])
    D, E, Fp = A + up, B + up, C + up
    G = Fp + (D - Fp) / 4                      # FG=1 (DF=4)
    H = (B + C) / 2
    u, v, h = sp.symbols("u v h", real=True)   # P=B+u(G-B), Q=H+v(I-H), I=(0,0,h)
    I = sp.Matrix([0, 0, h])
    P = B + u * (G - B); Q = H + v * (I - H)
    sol = sp.solve([P[0] - Q[0], P[1] - Q[1]], [u, v], dict=True)[0]
    PQ = sp.simplify((P[2] - Q[2]).subs(sol))
    hs = [r for r in sp.solve(sp.Eq(sp.Abs(PQ), sp.Rational(26, 7)), h) if 0 <= r <= 8]
    ok1 = len(hs) == 1 and sp.simplify(hs[0] - sp.sympify(md_answer(3, "1"))) == 0
    hv = hs[0]
    Pn = [sp.N(c, 30) for c in P.subs(sol).subs(h, hv)]
    In = I.subs(h, hv)
    pq = abs(float(PQ.subs(h, hv)))
    # (3-2) 평면 EGI 위 영역: 거리(직선 PQ, 연직) = 수평거리 r ∈ [1, √7/2], 삼각형 EGI 안.
    # 정사영 영역 넓이를 P' 중심 극좌표 적분으로 구하고 cosθ(법선벡터)로 나눈다.
    Ex, Gx, Ix = [(mp.mpf(str(sp.N(p[0], 30))), mp.mpf(str(sp.N(p[1], 30)))) for p in (E, G, In)]
    c = (mp.mpf(str(Pn[0])), mp.mpf(str(Pn[1])))
    tri = [Ex, Gx, Ix]
    def exit_dist(phi):
        d = (mp.cos(phi), mp.sin(phi)); best = mp.inf
        for i in range(3):
            p1, p2 = tri[i], tri[(i + 1) % 3]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            den = d[0] * ey - d[1] * ex
            if abs(den) < mp.mpf(10)**-25:
                continue
            wx, wy = p1[0] - c[0], p1[1] - c[1]
            s_ = (wx * ey - wy * ex) / den; w_ = (wx * d[1] - wy * d[0]) / den
            if s_ > mp.mpf(10)**-20 and -1e-20 <= w_ <= 1 + 1e-20:
                best = min(best, s_)
        return best if best < mp.inf else mp.mpf(0)
    def _unused(phi):     # 짧게 나아간 점이 삼각형 안인가
        q = (c[0] + mp.mpf(10)**-8 * mp.cos(phi), c[1] + mp.mpf(10)**-8 * mp.sin(phi))
        sg = []
        for i in range(3):
            p1, p2 = tri[i], tri[(i + 1) % 3]
            sg.append((p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0]))
        return all(x_ >= 0 for x_ in sg) or all(x_ <= 0 for x_ in sg)
    r1, r2 = mp.mpf(1), mp.sqrt(7) / 2
    def dens(phi):
        dd = exit_dist(phi)
        return (min(dd, r2)**2 - min(dd, r1)**2) / 2
    # P' 는 변 EG(정사영 BG') 위 → 안쪽 방향은 P'→E 방향부터 I 쪽으로 π 만큼
    ang = lambda p: mp.atan2(p[1] - c[1], p[0] - c[0])
    aE, aI = ang(Ex), ang(Ix)
    dI = (aI - aE) % (2 * mp.pi)
    sgn = 1 if dI < mp.pi else -1
    lo, mid, hi = aE, aE + sgn * (dI if sgn == 1 else 2 * mp.pi - dI), aE + sgn * mp.pi
    proj = abs(mp.quad(dens, [lo, mid, hi]))
    n_vec = (G - E).cross(In - E)
    cos_t = abs(float(n_vec[2])) / float(n_vec.norm())
    area = proj / cos_t
    ans = sp.N(sp.sympify(md_answer(3, "2")), 30)
    ok2 = abs(area - mp.mpf(str(ans))) < 1e-6
    # 삼각형 PQR 넓이 = 1/2·PQ·r 인지 임의 점에서 확인
    Rt = sp.Matrix([1.1, 0.7, 7.5]); Pv = sp.Matrix(Pn); Qv = Pv - sp.Matrix([0, 0, pq])
    T_area = float((Qv - Pv).cross(Rt - Pv).norm()) / 2
    rr = float(sp.sqrt((Rt[0] - Pv[0])**2 + (Rt[1] - Pv[1])**2))
    ok2 = ok2 and abs(T_area - pq * rr / 2) < 1e-9
    report(3, ok1 and ok2, f"AI={hv} PQ={pq:.6f} proj_area={mp.nstr(proj, 12)} (3π/8={mp.nstr(3*mp.pi/8, 12)}) cosθ={cos_t:.10f} area={mp.nstr(area, 12)}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
