#!/usr/bin/env python3
"""SGU-G23B (서강대 2023학년도 수시 논술 자연계열 2차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import random
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SGU-G23B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str) -> dict:
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = re.split(r"[:=]", seg, maxsplit=1)
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def circle_on_s(beta, px):
    """x축 위 접점 P(px,0), 점 (1,0)을 지나는 경사각 beta 직선 s 에 접하는 원(중심 x좌표 < 1, b>0): (r, 중심각 PCQ)."""
    r = sp.symbols("r")
    # 직선 s: sinβ·x − cosβ·y − sinβ = 0, 중심 (px, r) 은 s 의 왼쪽(값이 음수) 쪽
    val = sp.sin(beta) * px - sp.cos(beta) * r - sp.sin(beta)
    rr = sp.solve(sp.Eq(-val, r), r)[0]
    C = sp.Matrix([px, rr])
    nrm = sp.Matrix([sp.sin(beta), -sp.cos(beta)])           # 단위 법선
    Q = C - (nrm.dot(C) - sp.sin(beta)) * nrm                  # C 에서 s 에 내린 수선의 발
    CP, CQ = sp.Matrix([px, 0]) - C, Q - C
    ang = sp.acos(sp.simplify(CP.dot(CQ) / (CP.norm() * CQ.norm())))
    return sp.simplify(rr), ang, Q


def p01():
    # (1)
    al, be = sp.pi / 12, sp.pi / 4
    A = sp.Matrix([sp.cos(al), sp.sin(al)]); B = A + sp.Matrix([sp.cos(be), sp.sin(be)])
    perp = sp.simplify(-B[0] / B[1])
    ok1 = sp.simplify(perp - sp.sympify(md_answer(1, "1"))) == 0
    # (2) β=π/3: 중심 (a,b) 가 x축·s 에 접하는 조건 → 자취
    a, b = sp.symbols("a b", real=True)
    sols = [sp.solve(sp.Eq(e / 2, b), a)[0] for e in (sp.sqrt(3) * a - b - sp.sqrt(3), -(sp.sqrt(3) * a - b - sp.sqrt(3)))]
    locus = [so for so in sols if (so.subs(b, 1) < 1)]                       # a<1 인 가지
    ok_loc = len(locus) == 1
    pr = md_parts(1, "2")
    line_md = sp.sympify(pr["line"].split("=", 1)[1], locals={"x": x})       # y = ... 의 우변
    # 자취 a=φ(b) 를 y=… 꼴로: b = line(a)
    ok_line = all(abs(float(line_md.subs(x, locus[0].subs(b, bv)) - bv)) < 1e-12 for bv in (0.1, 0.5, 2))
    # 삼각형(x축, s, y=5x/12)의 내접원: 2·넓이/둘레
    V1, V2 = sp.Matrix([0, 0]), sp.Matrix([1, 0])
    xs = sp.solve(sp.Eq(sp.Rational(5, 12) * x, sp.sqrt(3) * (x - 1)), x)[0]
    V3 = sp.Matrix([xs, sp.Rational(5, 12) * xs])
    area = sp.Abs((V2 - V1)[0] * (V3 - V1)[1] - (V2 - V1)[1] * (V3 - V1)[0]) / 2
    per = (V2 - V1).norm() + (V3 - V2).norm() + (V1 - V3).norm()
    rin = sp.nsimplify(sp.simplify(2 * area / per))
    ok2 = ok_loc and ok_line and abs(float(rin - sp.sympify(pr["r"]))) < 1e-12
    # (3) β=π/3, P(n/100,0)
    nn = sp.symbols("n", positive=True)
    r3, ang3, _ = circle_on_s(sp.pi / 3, nn / 100)
    angs = [float(ang3.subs(nn, k)) for k in (1, 37, 99)]
    ln = r3 * sp.pi / 3
    ok_ang = all(abs(v - math.pi / 3) < 1e-12 for v in angs)
    p3 = md_parts(1, "3")
    ln_md = sp.sympify(p3["l_n"], locals={"n": nn})
    tot = sp.summation(ln.subs(nn, n), (n, 1, 99))
    ok3 = ok_ang and sp.simplify(ln - ln_md) == 0 and sp.simplify(tot - sp.sympify(p3["sum"])) == 0
    # (4) P(1/4,0), 일반 β: 중심각이 β 임을 수치 확인 후 S=½r²β 극한
    bb = sp.symbols("beta", positive=True)
    r4, ang4, _ = circle_on_s(bb, sp.Rational(1, 4))
    ok_ang4 = all(abs(float(ang4.subs(bb, v)) - v) < 1e-10 for v in (0.05, 0.4, 1.2, 1.5))
    S = r4**2 * bb / 2
    lim4 = sp.limit(S * sp.tan(bb), bb, 0, "+")
    ok4 = ok_ang4 and sp.simplify(lim4 - sp.sympify(md_answer(1, "4"))) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"perp={perp} r_in={rin} sum={sp.simplify(tot)} lim={lim4}")


def p02():
    # (1) 좌·우극한
    d = sp.sqrt(x**2 + (sp.exp(x) - 1)**2)
    rp, lm = sp.limit(d / x, x, 0, "+"), sp.limit(d / x, x, 0, "-")
    p1 = md_parts(2, "1")
    ok1 = sp.simplify(rp - sp.sympify(p1["right"])) == 0 and sp.simplify(lm - sp.sympify(p1["left"])) == 0 \
        and (rp != lm) == (p1["limit"] == "DNE")
    # (2) [증명-수치확인] 접선식 비교 + 무작위 (p,c), 격자 x 에서 l(x) >= x^p
    p, c = sp.symbols("p c", positive=True)
    l = sp.diff(x**p, x).subs(x, c) * (x - c) + c**p
    l_md = sp.sympify(md_answer(2, "2").split(";")[0].split("=", 1)[1], locals={"x": x, "p": p, "c": c})
    ok_l = sp.simplify(sp.expand(l - l_md)) == 0
    random.seed(1)
    worst = min(
        pv * cv**(pv - 1) * xv + (1 - pv) * cv**pv - xv**pv
        for pv, cv in [(random.uniform(0.01, 0.99), random.uniform(0.01, 5)) for _ in range(20)]
        for xv in [k * 10 / 1000 for k in range(1001)])
    ok2 = ok_l and worst >= -1e-12
    # (3) h(c)=∫0^1 (l - x^p) dx 를 c 로 최소화(여러 p)
    h = sp.integrate(sp.expand(l - x**p), (x, 0, 1))
    crit = sp.solve(sp.Eq(sp.diff(h, c), 0), c)
    c_md = sp.sympify(md_answer(2, "3"))
    ok3 = crit == [c_md]
    for pv in (sp.Rational(1, 10), sp.Rational(1, 2), sp.Rational(9, 10)):
        hf = sp.lambdify(c, h.subs(p, pv))
        cs = [k / 1000 for k in range(1, 5001)]
        cbest = min(cs, key=hf)
        ok3 = ok3 and abs(cbest - float(c_md)) < 2e-3
    # (4) c=1/e: S=∫0^{1/e}, R=∫_{1/e}^1
    le = l.subs(c, sp.exp(-1))
    S = sp.integrate(sp.expand(le - x**p), (x, 0, sp.exp(-1)))
    SR = sp.integrate(sp.expand(le - x**p), (x, 0, 1))
    lim4 = sp.limit(sp.simplify(SR / S), p, 0, "+")
    mp.mp.dps = 40
    Sf, SRf = sp.lambdify(p, S, "mpmath"), sp.lambdify(p, SR, "mpmath")
    num = SRf(mp.mpf("1e-12")) / Sf(mp.mpf("1e-12"))
    ok4 = sp.simplify(lim4 - sp.sympify(md_answer(2, "4"))) == 0 and abs(num - mp.e**2 + 2 * mp.e) < 1e-9
    report(2, ok1 and ok2 and ok3 and ok4, f"lim±=({rp},{lm}) min(l-x^p)={worst:.2e} c*={crit} lim4={lim4}")


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
