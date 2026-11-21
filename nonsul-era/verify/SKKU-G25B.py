#!/usr/bin/env python3
"""SKKU-G25B (성균관대 2025학년도 수시 논술우수 수리형 2교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import sys

import numpy as np
import sympy as sp

PREFIX = "SKKU-G25B"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_set(no, sub, loc=None):
    v = md_answer(no, sub).strip().strip("{}")
    return [sp.sympify(p, locals=loc or {}) for p in v.split(",")]


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def area(tv):
    """넓이 g(t) = ∫_t^{t+1} |3x^2-3| dx (부호 변화점 ±1 에서 나눠 정확 계산)."""
    pts = sorted({tv, tv + 1} | {p for p in (-1, 1) if tv < p < tv + 1})
    return sum(sp.Abs(sp.integrate(3 * x**2 - 3, (x, a, b))) for a, b in zip(pts, pts[1:]))


def p01():
    ok1 = sp.simplify(area(sp.Rational(-3, 2)) - sp.sympify(md_answer(1, "1"))) == 0
    G = sp.sympify(md_answer(1, "2"), locals={"t": t})
    samples = [sp.Rational(k, 7) for k in range(-21, 1)] + [sp.Rational(-5, 2), sp.Rational(-3, 2), -2, -1]
    ok2 = all(sp.simplify(area(v) - G.subs(t, v)) == 0 for v in samples)
    # (iii) 최솟값: 구간별 식으로 임계점·끝점 비교 (독립: 구간 [-3,0] 에서 직접 탐색)
    cands = [sp.Integer(-3), sp.Integer(0), sp.Integer(-2), sp.Integer(-1)]
    for lo, hi in ((-3, -2), (-2, -1), (-1, 0)):
        mid = sp.Rational(lo + hi, 2)
        piece = G.subs(t, mid)  # 값 확인용
        expr = [e for e, c in G.args if c.subs(t, mid)][0] if isinstance(G, sp.Piecewise) else G
        for r in sp.solve(sp.diff(expr, t), t):
            if r.is_real and lo < r < hi:
                cands.append(r)
    mn = min(cands, key=lambda v: float(area(v)))
    mval = sp.nsimplify(sp.simplify(area(mn)))
    grid = min(float(area(sp.Rational(k, 400))) for k in range(-1200, 1))
    ok3 = sp.simplify(mval - sp.sympify(md_answer(1, "3"))) == 0 and grid >= float(mval) - 1e-12
    report(1, ok1 and ok2 and ok3, f"g(-3/2)={area(sp.Rational(-3, 2))} min={mval} at t={mn}")


def square(av):
    """중심 (av,0), 대각선 AC 기울기 av, 대각선 길이 4, 반시계, D 의 y>0 인 꼭짓점 A,B,C,D(수치)."""
    u = np.array([1.0, av]) / np.hypot(1, av)          # AC 방향 단위벡터
    w = np.array([-u[1], u[0]])                          # 90° 회전
    c = np.array([av, 0.0])
    for s in (1, -1):
        A = c - 2 * s * u; C = c + 2 * s * u
        for r in (1, -1):
            B = c - 2 * r * w; D = c + 2 * r * w
            P = [A, B, C, D]
            area2 = sum(P[i][0] * P[(i + 1) % 4][1] - P[(i + 1) % 4][0] * P[i][1] for i in range(4))
            if area2 > 0 and D[1] > 1e-12:
                return P
    return None


def p02():
    a = sp.symbols("a", real=True)
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "1").replace(" ", "").split(";"))
    F = sp.sympify(parts["f(a)"], locals={"a": a}); Gd = sp.sympify(parts["g(a)"], locals={"a": a})
    ok1 = True
    for av in (-3.1, -1, -0.2, 0.0, 0.5, 1, 2.7):
        P = square(av)
        ok1 &= P is not None and abs(P[0][1] - float(F.subs(a, av))) < 1e-12 and abs(P[3][1] - float(Gd.subs(a, av))) < 1e-12
    # (ii) |f-g|=1 을 정확히 풀기
    sol2 = [s for s in sp.solve(sp.Eq((F - Gd)**2, 1), a) if abs(abs(float((F - Gd).subs(a, s))) - 1) < 1e-12]
    md2 = md_set(2, "2")
    ok2 = len(sol2) == len(md2) and all(any(sp.simplify(u - v) == 0 for v in sol2) for u in md2)
    # (iii) 원점과 직선 y=a(x-a) 거리 2 → a^2 값
    sol3 = {sp.simplify(s**2) for s in sp.solve(sp.Eq(a**4, 4 * (a**2 + 1)), a) if s.is_real}
    md3 = md_set(2, "3")
    ok3 = len(sol3) == len(md3) and all(any(sp.simplify(u - v) == 0 for v in sol3) for u in md3)
    # (iv) 수열을 수치로 생성해 누적합으로 최소 m
    terms = []
    for k in range(0, 200):                              # (0, 800] 탐색: sin(pi a/2)=±√3/2
        for base in (sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(4, 3), sp.Rational(5, 3)):
            val = 2 * base + 4 * k                        # pi a/2 = base*pi + 2k pi
            terms.append(val)
    terms = sorted(v for v in set(terms) if v > 0)
    terms = [v for v in terms if abs(abs(float(F.subs(a, 2 * sp.sin(sp.pi * v / 2)))) - np.sqrt(3)) < 1e-9]
    # 누락 확인: 격자 탐색으로 해 개수 비교(0<a≤8)
    xs = np.linspace(1e-6, 8, 800001)
    vals = np.abs(-2 * (2 * np.sin(np.pi * xs / 2)) / np.sqrt(1 + (2 * np.sin(np.pi * xs / 2))**2)) - np.sqrt(3)
    ncross = int(np.sum(np.sign(vals[:-1]) != np.sign(vals[1:])))
    S, m = 0, None
    for i, v in enumerate(terms, 1):
        S += v
        if S > 2025:
            m = i
            break
    ok4 = ncross == sum(1 for v in terms if v <= 8) and m == int(md_answer(2, "4"))
    report(2, ok1 and ok2 and ok3 and ok4, f"(ii)={sol2} (iii)a^2={sol3} m={m}")


def p03():
    A = sp.symbols("A", real=True)
    def fcount(av):
        return len(set(sp.solve(sp.Eq(-x**2, av * x + av), x)))
    s1 = sum(fcount(v) for v in range(0, 6))
    ok1 = s1 == int(md_answer(3, "1"))
    # (ii) g = x^4 + p x^3 + q x^2 + r x + s 의 일반형에서 조건을 연립(독립 유도)
    p, q, r, s = sp.symbols("p q r s", real=True)
    g = x**4 + p * x**3 + q * x**2 + r * x + s
    # 연속: f 는 x=0,4 에서만 불연속(값 1, 양옆 2·0) → 2g(0)=g(0)=0·…: g(0)=0, g(4)=0
    # 미분가능(4): 좌미분계수 0, 우미분계수 2g'(4)
    eqs = [g.subs(x, 0), g.subs(x, 4), sp.diff(g, x).subs(x, 4)]
    sol = sp.solve(eqs, [q, r, s], dict=True)[0]
    gg = sp.expand(g.subs(sol))                         # p 하나 남음
    fpw = lambda v: 2 if (v < 0 or v > 4) else (1 if v in (0, 4) else 0)
    h = lambda v: fpw(v) * gg.subs(x, v)
    val = sp.simplify(h(-1) + 3 * h(2) + 5 * h(5))
    ok2 = val.free_symbols == set() and {val} == set(md_set(3, "2"))
    # (iii) 정수 p 전수: g 실근 모두 정수(허근 없음), g'+5x^3-20x^2 실근 모두 정수
    cnt = 0
    for pv in range(-200, 201):
        G = gg.subs(p, pv)
        rts = sp.Poly(G, x).all_roots()
        if any(not rt.is_real for rt in rts) or any(not rt.is_integer for rt in rts):
            continue
        H = sp.diff(G, x) + 5 * x**3 - 20 * x**2
        hr = sp.Poly(H, x).real_roots()
        if all(rt.is_integer for rt in hr):
            cnt += 1
    ok3 = cnt == int(md_answer(3, "3"))
    report(3, ok1 and ok2 and ok3, f"sum f={s1} value={val} count={cnt}")


if __name__ == "__main__":
    for name in sorted(n for n in dir() if n.startswith("p") and n[1:].isdigit()):
        globals()[name]()
    n_ok = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {n_ok}/{len(RESULTS)} PASS")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
