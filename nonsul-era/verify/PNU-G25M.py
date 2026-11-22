#!/usr/bin/env python3
"""PNU-G25M (부산대 2025학년도 수시 지역인재전형(논술) 의·약학계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지. 문항 YAML 의 answer 를 읽어 독립 재계산값과 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "PNU-G25M"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


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
    # f'(x)=k x(x+1)(x-a) (k>0, a>1) 꼴에서 미지수 k, a, c(=f(0)) 를 조건으로 결정한다.
    k, a = sp.symbols("k a", positive=True)
    fp = k * t * (t + 1) * (t - a)
    C = sp.Symbol("C", real=True)             # f(0)
    f = sp.integrate(fp, (t, 0, x))           # f(x) - f(0) = ∫_0^x f'
    F = lambda v: f.subs(x, v) + C            # noqa: E731
    # g(x)=∫_1^x |f'| (x>=-1). 부호: (-1,0) 양, (0,1)⊂(0,a) 음
    g_m1 = -(sp.integrate(fp, (t, -1, 0)) - sp.integrate(fp, (t, 0, 1)))   # g(-1) = -∫_{-1}^{1}|f'|
    g_0 = sp.integrate(fp, (t, 0, 1))                                      # g(0) = -∫_0^1|f'| = ∫_0^1 f'
    sols = sp.solve([sp.Eq(F(-1), -18), sp.Eq(g_m1, -18), sp.Eq(g_0, -13)], [k, a, C], dict=True)
    sols = [s for s in sols if s[k] > 0 and s[a] > 1]
    ok_unique = len(sols) == 1
    s = sols[0]
    fx = sp.expand((f + C).subs(s))
    kv, av = s[k], s[a]
    # 조건 수치 확인: g 정의(절댓값 적분), 미분가능성, g>=-18, g'(0)=0, g(0)=-13, g'(a)=0
    fl = sp.lambdify(x, fx, "mpmath")
    fpl = sp.lambdify(x, sp.diff(fx, x), "mpmath")
    mp.mp.dps = 30
    brk = [-1, 0, float(av)]
    def g(v):
        v = mp.mpf(v)
        if v < -1:
            return fl(v)
        lo, hi, sgn = (1, v, 1) if v >= 1 else (v, 1, -1)
        pts = [lo] + [p for p in brk if lo < p < hi] + [hi]
        return sgn * mp.quad(lambda u: abs(fpl(u)), pts)
    grid = [-4 + i * 0.01 for i in range(801)]
    ok_min = min(g(v) for v in grid) >= -18 - 1e-12 and abs(g(-1) + 18) < 1e-15
    h = mp.mpf("1e-8")
    ok_diff = abs((g(-1) - g(-1 - h)) / h - (g(-1 + h) - g(-1)) / h) < 1e-6
    ok_c = abs(g(0) + 13) < 1e-15 and abs(fpl(0)) < 1e-20 and abs(fpl(float(av))) < 1e-20 and av > 1
    v1 = fx.subs(x, 0)
    v3 = g(float(av) + 1) - 2 * g(float(av) - 1)
    ok1 = sp.simplify(v1 - sp.sympify(md_answer(1, "1"))) == 0
    ok2 = sp.simplify(av - sp.sympify(md_answer(1, "2"))) == 0
    ok3 = abs(v3 - float(sp.sympify(md_answer(1, "3")))) < 1e-12
    report(1, all([ok_unique, ok_min, ok_diff, ok_c, ok1, ok2, ok3]),
           f"k={kv} a={av} f={fx} f(0)={v1} g(a+1)-2g(a-1)={mp.nstr(v3, 15)}")


def _clip(subject, clipper):
    """볼록 다각형 교집합(Sutherland–Hodgman). clipper 는 반시계 방향."""
    out = subject
    n = len(clipper)
    for i in range(n):
        A, B = clipper[i], clipper[(i + 1) % n]
        inside = lambda p: (B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (p[0] - A[0]) >= 0  # noqa: E731
        def inter(p, q):
            x1, y1 = p; x2, y2 = q; x3, y3 = A; x4, y4 = B
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            tt = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + tt * (x2 - x1), y1 + tt * (y2 - y1))
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(inter(s, e))
                out.append(e)
            elif inside(s):
                out.append(inter(s, e))
            s = e
    return out


def _area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def _config(th):
    sq = [(mp.cos(th + j * mp.pi / 2), mp.sin(th + j * mp.pi / 2)) for j in range(4)]       # P,P1,P2,P3 (반시계)
    tr = [(1 + mp.cos(th + j * 2 * mp.pi / 3), mp.sin(th + j * 2 * mp.pi / 3)) for j in range(3)]  # Q,Q1,Q2
    return sq, tr


def _dist(p, a, b):
    return abs((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / mp.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


def p02():
    mp.mp.dps = 40
    th = sp.Symbol("th", real=True)
    # [2-1] 문제에 주어진 d1, d2 식(증명 대상)을 좌표로 계산한 거리와 비교 [증명-수치확인]
    d1_claim = sp.lambdify(th, (sp.Rational(3, 2) + sp.sqrt(3) / 2 - sp.cos(th) - sp.sin(th)) / sp.sqrt(2), "mpmath")
    d2_claim = sp.lambdify(th, sp.Abs(sp.sin(th - sp.pi / 6)), "mpmath")
    grid = [mp.pi / 9 + (mp.pi / 3 - mp.pi / 9) * i / 400 for i in range(401)]
    e1 = e2 = 0
    for v in grid:
        sq, tr = _config(v)
        P, P3 = sq[0], sq[3]
        Q, Q1 = tr[0], tr[1]
        e1 = max(e1, abs(_dist(Q1, P, P3) - d1_claim(v)))
        e2 = max(e2, abs(_dist(P, Q, Q1) - d2_claim(v)))
    ok1 = e1 < 1e-30 and e2 < 1e-30
    # [2-2] YAML 의 S1, S2 식을 실제 다각형 교집합 넓이와 비교, 좌우 도함수 극한 비교
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "2").replace(" ", "").split(";"))
    loc = {"th": th}
    S1 = sp.sympify(parts["S1"], locals=loc)
    S2 = sp.sympify(parts["S2"], locals=loc)
    L = sp.sympify(parts["limit"])
    S1l, S2l = sp.lambdify(th, S1, "mpmath"), sp.lambdify(th, S2, "mpmath")
    Sgeo = lambda v: _area(_clip(*_config(v)))  # noqa: E731
    e3 = 0
    for v in grid:
        if abs(v - mp.pi / 6) < 1e-30:
            continue
        ref = S1l(v) if v < mp.pi / 6 else S2l(v)
        e3 = max(e3, abs(Sgeo(v) - ref))
    ok2a = e3 < 1e-25
    # 실제 넓이 함수의 수치 한쪽 도함수(θ→π/6±)
    c = mp.pi / 6
    hs = [mp.mpf(10) ** (-k) for k in (6, 8, 10)]
    left = [(Sgeo(c - h) - Sgeo(c - 2 * h)) / h for h in hs]
    right = [(Sgeo(c + 2 * h) - Sgeo(c + h)) / h for h in hs]
    lim_l = sp.limit(sp.diff(S1, th), th, sp.pi / 6, "-")
    lim_r = sp.limit(sp.diff(S2, th), th, sp.pi / 6, "+")
    ok2b = sp.simplify(lim_l - L) == 0 and sp.simplify(lim_r - L) == 0 and \
        abs(left[-1] - float(L)) < 1e-6 and abs(right[-1] - float(L)) < 1e-6
    report(2, ok1 and ok2a and ok2b,
           f"[증명-수치확인] d1,d2 최대오차 {mp.nstr(e1, 3)},{mp.nstr(e2, 3)}; S 식 오차 {mp.nstr(e3, 3)}; "
           f"lim- ={sp.nsimplify(lim_l)} lim+ ={sp.nsimplify(lim_r)} 수치 {mp.nstr(left[-1], 10)}/{mp.nstr(right[-1], 10)}")


def p03():
    a, b, s, tt = sp.symbols("a b s tt", real=True)
    A = sp.Matrix([0, 3]); B = sp.Matrix([3 * sp.sqrt(3) / 2, -sp.Rational(3, 2)])
    v = A + B                                   # OA+OB, Q = B + s v (s=1/k≠0)
    P = sp.Matrix([a, b]); AP = P - A
    # (나): AP·(Q-P)=0 에 Q=B+s v 대입 → s 의 일차식 α s + β = 0
    eq = sp.expand(AP.dot(B + s * v - P))
    al, be = eq.coeff(s, 1), eq.subs(s, 0)
    # [3-1] Q 가 없을 조건: α=0, β≠0 (또는 해가 s=0 뿐: β=0 → Q=B → AP·BP=0 이라 (가)에서 제외)
    cand = sp.solve([al, a**2 + b**2 - 4], [a, b], dict=True)
    good = []
    for c in cand:
        apbp = sp.simplify((P - A).dot(P - B).subs(c))
        if sp.simplify(apbp) != 0 and sp.simplify(be.subs(c)) != 0:
            good.append((sp.nsimplify(sp.radsimp(c[a])), sp.nsimplify(sp.radsimp(c[b]))))
    md = sp.sympify(md_answer(3, "1").replace("{", "[").replace("}", "]"))
    ok1 = len(good) == len(md) == 2 and all(
        any(sp.simplify(g[0] - m[0]) == 0 and sp.simplify(g[1] - m[1]) == 0 for m in md) for g in good)
    # [3-2] P=(2cosφ,2sinφ) 전수 격자 → Q=B+s v, |OA+OQ| 최소(조건 (가) 위반·Q 없음 제외)
    mp.mp.dps = 30
    alf = sp.lambdify((a, b), al, "mpmath"); bef = sp.lambdify((a, b), be, "mpmath")
    def val(ph):
        pa, pb = 2 * mp.cos(ph), 2 * mp.sin(ph)
        if abs(alf(pa, pb)) < 1e-20:
            return mp.inf
        sv = -bef(pa, pb) / alf(pa, pb)
        if abs(sv) < 1e-20:
            return mp.inf
        Qx = float(B[0]) + sv * float(v[0]); Qy = float(B[1]) + sv * float(v[1])
        return mp.sqrt(Qx**2 + (Qy + 3)**2)
    N = 20000
    ph0 = min((2 * mp.pi * i / N for i in range(N)), key=val)
    phm = mp.findroot(lambda p: mp.diff(val, p), ph0)
    m = val(phm)
    pa, pb = 2 * mp.cos(phm), 2 * mp.sin(phm)
    apbp = pa * (pa - float(B[0])) + (pb - 3) * (pb - float(B[1]))
    ok2 = abs(m - float(sp.sympify(md_answer(3, "2")))) < 1e-12 and abs(apbp) > 1e-3
    report(3, ok1 and ok2, f"P(3-1)={good}; min|OA+OQ|={mp.nstr(m, 15)} at P=({mp.nstr(pa, 8)},{mp.nstr(pb, 8)})")


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
