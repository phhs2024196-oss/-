#!/usr/bin/env python3
"""SJU-G15A (세종대 2015학년도 수시 논술[A] 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G15A"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    P = sp.Matrix([0, -sp.sqrt(3), 5]); Q = sp.Matrix([1, 0, 3])
    L = 4 * sp.sqrt(2)
    c = (P - Q).norm() / 2; a = L / 2; b2 = sp.simplify(a**2 - c**2)
    X, Y = sp.symbols("X Y", real=True)
    ell = X**2 / a**2 + Y**2 / b2 - 1
    lhs = sp.sympify(md_answer(1, "1").split("=")[0], locals={"X": X, "Y": Y})
    ok1 = sp.simplify(lhs - 1 - ell) == 0
    # 평면 α(수직면) 위 좌표: 수평축 u (Q→P 수평 방향), 높이 z. X 단위벡터 = (P-Q)/|P-Q| 를 (u,z)로
    hu = sp.sqrt((P[0] - Q[0])**2 + (P[1] - Q[1])**2); hz = P[2] - Q[2]
    ex = sp.Matrix([hu, hz]) / sp.sqrt(hu**2 + hz**2)
    ey = sp.Matrix([-ex[1], ex[0]])                   # 왼쪽 위 (그림 2)
    zc = (P[2] + Q[2]) / 2
    zf = zc + X * ex[1] + Y * ey[1]                    # 높이
    Ysol = sp.solve(sp.Eq(zf, 0), Y)[0]
    A2 = parts(md_answer(1, "2"))
    ok2 = sp.simplify(Ysol - (sp.sympify(A2["a"]) * X + sp.sympify(A2["b"]))) == 0
    th = sp.symbols("th", real=True)
    hz_par = zf.subs({X: a * sp.cos(th), Y: sp.sqrt(b2) * sp.sin(th)})
    import mpmath as mp
    fz = sp.lambdify(th, hz_par, "mpmath")
    grid = [2 * mp.pi * i / 2000 for i in range(2000)]
    t0 = min(grid, key=fz)
    tm = mp.findroot(lambda s: mp.diff(fz, s), t0)
    ok3 = abs(fz(tm) - mp.mpf(float(sp.N(sp.sympify(md_answer(1, "3")), 30)))) < 1e-12
    report(1, ok1 and ok2 and ok3, f"a^2={a**2} b^2={b2} ell: Y={Ysol} min height={mp.nstr(fz(tm), 15)}")


def p02():
    a, b, c = sp.symbols("a b c")
    N = sp.Matrix([[a, 1], [b, c]]); E = sp.eye(2)
    sols = [s for s in sp.solve(list(N**3), [b, c], dict=True)]
    ok = len(sols) > 0
    for s in sols:
        Ns = N.subs(s)
        ok &= sp.simplify((E - Ns) * (E + Ns + Ns**2) - E) == sp.zeros(2)       # (2-1)
        ok &= sp.simplify((E + Ns + Ns**2) * (E - Ns) - E) == sp.zeros(2)
        ok &= sp.simplify(Ns.det()) == 0 and sp.simplify(s[b] - a * s[c]) == 0   # (2-3)
        ok &= sp.simplify(Ns**2) == sp.zeros(2)                                  # (2-4)
        # (2-2) NA=AN 인 A 의 일반형에서 md 답이 역행렬인지
        p, q, r, t_ = sp.symbols("p q r t_")
        A = sp.Matrix([[p, q], [r, t_]])
        cs = sp.solve(list(Ns * A - A * Ns), [r, t_], dict=True)[0]
        A = A.subs(cs)
        inv = sp.sympify(md_answer(2, "2"), locals={"E": E, "A": A, "N": Ns})
        ok &= sp.simplify((E - A * Ns) * inv - E) == sp.zeros(2) and sp.simplify(inv * (E - A * Ns) - E) == sp.zeros(2)
    report(2, ok, f"[증명-수치확인] N^3=O 해 {sols}; (E-N)^-1, det N=0, b=ac, N^2=O, (E-AN)^-1 확인")


def p03():
    x, a_, b_ = sp.symbols("x a b", real=True)
    f = x**3 - 6 * x**2 + 11 * x - 6
    poly = sp.Poly(sp.expand(f.subs(x, a_ + x) + f.subs(x, a_ - x) - 2 * b_), x)
    sol = sp.solve(poly.coeffs(), [a_, b_], dict=True)[0]
    ok1 = (sol[a_], sol[b_]) == tuple(sp.sympify(md_answer(3, "1")))
    # (3-2) 무작위 g(홀함수), h((a,0) 대칭) 로 수치 확인
    import random
    ok2 = True
    for _ in range(20):
        A = random.uniform(-3, 3); k = random.uniform(-2, 2)
        g = lambda t: t**3 + k * t
        h = lambda t: (t - A)**5 - 2 * (t - A)
        for _ in range(5):
            xv = random.uniform(-5, 5)
            ok2 &= abs(g(h(A + xv)) + g(h(A - xv))) < 1e-6 * (1 + abs(g(h(A + xv))))
    # (3-3) 실수 세제곱근으로 수치 적분
    import mpmath as mp
    cr = lambda u: mp.sign(u) * abs(u)**(mp.mpf(1) / 3)
    I = mp.quad(lambda t: 4 + cr(t**3 - 6 * t**2 + 11 * t - 6), [-1, 1, 2, 3, 5])
    ok3 = abs(I - int(md_answer(3, "3"))) < 1e-10
    report(3, ok1 and ok2 and ok3, f"(a,b)=({sol[a_]},{sol[b_]}) [증명-수치확인] 합성 대칭 100점 I={mp.nstr(I, 15)}")


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
