#!/usr/bin/env python3
"""HIU-G26 (홍익대 2026학년도 수시 논술 자연계열 서울) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G26"
WIDTH = 2
RESULTS = []
t = sp.symbols("t", real=True)
x, y = sp.symbols("x y", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    """'a = 1/2; b = sqrt(3)/2' 형식을 dict 로."""
    out = {}
    for seg in s.split(";"):
        k, v = seg.split("=", 1) if seg.count("=") == 1 or ":" not in seg else seg.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    def curve(A, B, th):
        return A * sp.cos(2 * sp.pi * t), B * sp.cos(2 * sp.pi * t + th)
    # (1) A=B=1, theta=0 : y-x 항등적으로 0, x 범위 [-1,1]
    X, Y = curve(1, 1, 0)
    ok1 = sp.simplify(Y - X) == 0 and md_answer(1, "1").replace(" ", "") == "y=x,-1<=x<=1"
    # (2) dy/dx - (a + b cot 2πt) == 0
    X, Y = curve(1, 1, sp.pi / 3)
    d = sp.diff(Y, t) / sp.diff(X, t)
    ab = kv(md_answer(1, "2"))
    a, b = sp.sympify(ab["a"]), sp.sympify(ab["b"])
    ok2 = all(abs(float((d - a - b * sp.cot(2 * sp.pi * t)).subs(t, v))) < 1e-12 for v in (0.1, 0.33, 0.71))
    # (3) 길이 + 회전 방향
    X, Y = curve(sp.sqrt(5), sp.sqrt(5), sp.pi / 2)
    p = sp.periodicity(X, t)
    speed = sp.sqrt(sp.diff(X, t)**2 + sp.diff(Y, t)**2)
    L = sp.integrate(sp.simplify(speed), (t, 0, p / 4))
    cross = sp.simplify(X * sp.diff(Y, t) - Y * sp.diff(X, t))      # <0 이면 시계 방향
    d3 = kv(md_answer(1, "3"))
    ok3 = sp.simplify(L - sp.sympify(d3["length"])) == 0 and (cross < 0) and d3["direction"] == "좌원편광"
    # (4) 타원 4x^2+9y^2=36, 접선, 넓이 최소
    X, Y = curve(3, 2, sp.pi / 2)
    ok_ell = sp.simplify(4 * X**2 + 9 * Y**2 - 36) == 0
    s = sp.symbols("s", positive=True)                                # P=(3cos s, 2 sin s), 0<s<pi/2
    x1, y1 = 3 * sp.cos(s), 2 * sp.sin(s)
    F = 4 * x**2 + 9 * y**2 - 36
    tan = sp.diff(F, x).subs({x: x1, y: y1}) * (x - x1) + sp.diff(F, y).subs({x: x1, y: y1}) * (y - y1)
    d4 = md_answer(1, "4")
    md_tan = sp.sympify(d4.split(";")[0].split(":", 1)[1].replace("=", "-(") + ")", locals={"x": x, "y": y})
    md_tan = md_tan.subs({sp.Symbol("x1"): x1, sp.Symbol("y1"): y1})
    ok_tan = sp.simplify(sp.expand(tan) / 2 - sp.expand(md_tan)) == 0   # 4x1x+9y1y-36 비례 확인
    mx = sp.solve(tan.subs(y, 0), x)[0]
    ny = sp.solve(tan.subs(x, 0), y)[0]
    S = sp.simplify(mx * ny / 2)
    crit = sp.solve(sp.diff(S, s), s)
    Smin = min(sp.nsimplify(sp.simplify(S.subs(s, c))) for c in crit if 0 < float(c) < float(sp.pi / 2))
    ok4 = ok_ell and ok_tan and sp.simplify(Smin - sp.sympify(d4.split(";")[1].split("=")[1])) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"a,b={a},{b} p={p} L={L} cross={cross} Smin={Smin}")


def p02():
    from fractions import Fraction as Fr
    def step(dist):
        nd = {}
        for xv, pr in dist.items():
            pa = Fr(1, xv + 2)
            nd[xv + 1] = nd.get(xv + 1, 0) + pr * pa
            nd[0] = nd.get(0, 0) + pr * (1 - pa)
        return nd
    d1 = step({0: Fr(1)}); d2 = step(d1); d3 = step(d2)
    md1 = {int(re.search(r"=(\d)\)", seg).group(1)): Fr(seg.split("=")[-1].strip())
           for seg in md_answer(2, "1").replace(" ", "").split(";")}
    ok1 = md1 == {k: v for k, v in d2.items() if v}
    p = d3[0]
    ok2 = Fr(md_answer(2, "2")) == p
    m = sp.symbols("m", positive=True)
    ps = sp.Rational(p.numerator, p.denominator)
    msol = sp.solve(sp.Eq(4 * m * ps * (1 - ps), 280), m)[0]
    ok3 = sp.Integer(md_answer(2, "3")) == msol
    # (4) 정규근사: P(2Y-3>=a)=0.5 → a = E(2Y-3). np, n(1-p) >= 5 확인
    E = 2 * msol * ps - 3
    ok4 = msol * ps >= 5 and msol * (1 - ps) >= 5 and sp.Integer(md_answer(2, "4")) == E
    # 참고: 정확한 이항확률 P(Y>=168)
    from math import comb
    pf = float(ps); n = int(msol)
    exact = sum(comb(n, k) * pf**k * (1 - pf)**(n - k) for k in range(168, n + 1))
    report(2, ok1 and ok2 and ok3 and ok4, f"X2={dict(d2)} P(X3=0)={p} m={msol} a={E} (exact P(Y>=168)={exact:.4f})")


def p03():
    r3 = sp.sqrt(3)
    # 격자: 한 변 1, 세로 변. 원점에서 위로 (0,1) 변, 아래로 왼/오 대각 (±√3/2,-1/2).
    up = [sp.Matrix([0, 1]), sp.Matrix([r3 / 2, -sp.Rational(1, 2)]), sp.Matrix([-r3 / 2, -sp.Rational(1, 2)])]
    dn = [-v for v in up]
    # 원점에서 변을 따라 최대 6번 이동해 닿는 격자 꼭짓점을 전수하고, 크기 √3(A)·3(B)인 위치벡터를 모은다
    pts = {(0, 0): 0}
    frontier = [(sp.Matrix([0, 0]), 0)]
    seen = {(sp.Integer(0), sp.Integer(0))}
    for _ in range(6):
        nf = []
        for p, parity in frontier:
            for e in (up if parity == 0 else dn):
                q = p + e
                key = (sp.nsimplify(q[0]), sp.nsimplify(q[1]))
                if key not in seen:
                    seen.add(key); nf.append((q, 1 - parity))
        frontier = nf
    A = {k for k in seen if sp.simplify(k[0]**2 + k[1]**2 - 3) == 0}
    B = {k for k in seen if sp.simplify(k[0]**2 + k[1]**2 - 9) == 0}
    def parse(s):
        return {tuple(sp.nsimplify(sp.sympify(c)) for c in m.groups())
                for m in re.finditer(r"\(([^,()]+(?:\([^)]*\))?[^,()]*),([^()]+(?:\([^)]*\))?[^()]*)\)", s)}
    a1 = md_answer(3, "1").replace(" ", "")
    mdA, mdB = parse(a1.split("B:")[0]), parse(a1.split("B:")[1])
    ok1 = len(A) == 6 and len(B) == 6 and mdA == A and mdB == B
    vecs = [(v, "A") for v in A] + [(v, "B") for v in B]
    F1 = (-r3 / 2, -sp.Rational(5, 2))
    def choose(F):
        pos = [(sp.nsimplify(v[0] * F[0] + v[1] * F[1]), typ, v) for v, typ in vecs]
        pos = [q for q in pos if q[0] > 0]
        mn = min(q[0] for q in pos)
        cands = [q for q in pos if q[0] == mn]
        cands.sort(key=lambda q: q[1])          # 동률이면 A 우선
        return cands[0], len(cands)
    (d, typ, u), nt = choose(F1)
    d2 = kv(md_answer(3, "2"))
    mu = parse(d2["u"])
    ok2 = mu == {u} and typ == "A" and sp.sympify(d2["dot"]) == d
    kmin = None
    for k in range(1, 50):
        F2 = (3 * F1[0] + r3 * k, 3 * F1[1])
        (d_, typ_, u_), _ = choose(F2)
        if (sp.simplify(u_[0] + u[0]) == 0 and sp.simplify(u_[1] + u[1]) == 0):
            kmin = k; break
    ok3 = kmin == int(md_answer(3, "3"))
    report(3, ok1 and ok2 and ok3, f"|A|={len(A)} |B|={len(B)} u={u}({typ}, 동률 {nt}개) dot={d} kmin={kmin}")


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
