#!/usr/bin/env python3
"""SJU-G17A (세종대 2017학년도 수시 논술고사[A] 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G17A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    f = (sp.exp(x) + sp.exp(-x)) / 2
    ds = sp.sqrt(1 + sp.diff(f, x)**2)
    L1 = sp.Integral(ds, (x, 0, 1)).evalf(30)
    ok1 = abs(L1 - sp.sympify(md_answer(1, "1")).evalf(30)) < 1e-20
    # (2) 수치로 길이 3 인 x>0 을 찾음
    import mpmath as mp
    mp.mp.dps = 30
    arc = lambda X: mp.quad(lambda s: mp.sqrt(1 + ((mp.e**s - mp.e**(-s)) / 2)**2), [0, X])
    xq = mp.findroot(lambda X: arc(X) - 3, 1.5)
    Q = sp.sympify(md_answer(1, "2"))
    ok2 = abs(xq - Q[0].evalf(30)) < 1e-20 and abs((mp.e**xq + mp.e**(-xq)) / 2 - Q[1].evalf(30)) < 1e-20
    # (3) 위치(길이) 함수 차의 근을 t>=0 에서 직접 셈(격자 부호변화 + 중근 검사)
    sP = sp.integrate(6*t**2 - 8*t + 14, (t, 0, t))
    sQ = 3 + sp.integrate(3*t**2 + 4*t + 5, (t, 0, t))
    h = sp.expand(sP - sQ)
    roots = [r for r in sp.Poly(h, t).nroots() if abs(sp.im(r)) < 1e-12 and sp.re(r) >= 0]
    ok3 = len(roots) == int(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"L={float(L1):.10f} Qx={float(xq):.10f} roots(t>=0)={[round(float(sp.re(r)),5) for r in roots]}")


def p02():
    PA = sp.Rational(1, 100)
    pD = sp.Rational(10, 100) * PA + sp.Rational(1, 100) * (1 - PA)
    ok1 = pD == sp.Rational(md_answer(2, "1"))
    def p3(p1, p2):
        return p1 * PA / (p1 * PA + (1 - p2) * (1 - PA))
    base = p3(sp.Rational(80, 100), sp.Rational(90, 100))
    ok2 = base == sp.Rational(md_answer(2, "2"))
    # (3) 0.01 단위 전수: p1 증가 i 단계, p2 감소 j 단계, 비용 100i-50j <= 1500, p3 >= base
    best = None
    for i in range(0, 21):
        for j in range(0, 91):
            if 100 * i - 50 * j <= 1500 and p3(sp.Rational(80 + i, 100), sp.Rational(90 - j, 100)) >= base:
                best = max(best or 0, sp.Rational(80 + i, 100))
    # 연속 변수로도 확인: 선형계획 최대 x
    X, Y = sp.symbols("X Y", nonnegative=True)
    cond = sp.solve(sp.Eq(p3(sp.Rational(8, 10) + X, sp.Rational(9, 10) - Y), base), Y)[0]  # 경계 y(x)
    xmax = sp.solve(sp.Eq(100 * X - 50 * cond, 15), X)[0]
    ok3 = best == sp.Rational(md_answer(2, "3")) and sp.Rational(8, 10) + xmax == best
    report(2, ok1 and ok2 and ok3, f"P(D)={pD} p3={base} p1max(grid)={best} p1max(cont)={sp.Rational(8,10)+xmax}")


def p03():
    al = sp.symbols("alpha", positive=True)
    A = sp.Matrix([0, 0, 0]); B = sp.Matrix([8, 0, 0])
    cA = sp.sqrt(3) / 3
    C = 6 * sp.Matrix([cA, sp.sqrt(1 - cA**2), 0])
    area = sp.simplify(sp.Rational(1, 2) * (B - A).cross(C - A).norm())
    ok1 = sp.simplify(area - sp.sympify(md_answer(3, "1"))) == 0
    # D=(dx,dy,dz): |AD|=3√2, AD·AB/(|AD||AB|)=α, AD·AC/(|AD||AC|)=√6/3
    dx = 3 * sp.sqrt(2) * al
    dy = sp.symbols("dy", real=True)
    dy = sp.solve(sp.Eq((dx * C[0] + dy * C[1]) / (3 * sp.sqrt(2) * 6), sp.sqrt(6) / 3), dy)[0]
    dz2 = 18 - dx**2 - dy**2
    P = sp.Matrix([dx, dy, 0])
    uC = C / C.norm()
    AR = sp.simplify(P.dot(uC))                          # R = P 의 AC 위 정사영
    ok2 = sp.simplify(AR - sp.sympify(md_answer(3, "2"))) == 0
    PQ = sp.simplify(dy)                                  # AB = x축 → PQ = P 의 y좌표
    ok3 = sp.simplify(PQ - sp.sympify(md_answer(3, "3"), locals={"alpha": al})) == 0
    # (4) 부피 = (1/3)·넓이·높이 = 4√15
    sols = sp.solve(sp.Eq(sp.Rational(1, 3) * area * sp.sqrt(dz2), 4 * sp.sqrt(15)), al)
    def inside(a):
        Pp = [float(dx.subs(al, a)), float(dy.subs(al, a))]
        pts = [(0, 0), (8, 0), (float(C[0]), float(C[1]))]
        s = []
        for k in range(3):
            (x1, y1), (x2, y2) = pts[k], pts[(k + 1) % 3]
            s.append((x2 - x1) * (Pp[1] - y1) - (y2 - y1) * (Pp[0] - x1))
        return all(v > 1e-12 for v in s) or all(v < -1e-12 for v in s)
    good = [s for s in sols if inside(s)]
    ok4 = len(good) == 1 and sp.simplify(good[0] - sp.sympify(md_answer(3, "4"))) == 0
    report(3, ok1 and ok2 and ok3 and ok4, f"area={area} AR={AR} PQ={PQ} roots={sols} inside={good}")


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
