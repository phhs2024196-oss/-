#!/usr/bin/env python3
"""HIU-G23B (홍익대 2023학년도 수시 논술 자연계열 서울 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G23B"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {a.strip(): b.strip() for a, b in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    r = sp.symbols("r", positive=True)
    X = [sp.Matrix([4, 4]), sp.Matrix([5, 0]), sp.Matrix([2, 3]), sp.Matrix([1, 1])]
    c = sum(X, sp.zeros(2, 1)) / 4
    D = [sp.sqrt((xi - c).dot(xi - c)) for xi in X]          # 중심까지 거리
    # (1) S4(r)=Σ(D_i - r)^2 최소
    S = sum((Di - r) ** 2 for Di in D)
    r1 = sp.solve(sp.diff(S, r), r)[0]
    ok1 = sp.simplify(r1 - sp.sympify(md_answer(1, "1"))) == 0 and sp.diff(S, r, 2) > 0
    # (2) M4(r)=max|D_i - r| 를 격자 탐색 후 정밀 비교
    import mpmath as mp
    mp.mp.dps = 30
    Dn = [mp.mpf(str(sp.N(Di, 40))) for Di in D]
    M = lambda rv: max(abs(d - rv) for d in Dn)
    grid = [mp.mpf(i) / 10000 for i in range(1, 50001)]
    rg = min(grid, key=M)
    r2 = (min(Dn) + max(Dn)) / 2                              # 최대거리 최소화: 양 끝 거리의 중점
    r2md = mp.mpf(str(sp.N(sp.sympify(md_answer(1, "2")), 40)))
    ok2 = abs(rg - r2md) < 2e-4 and abs(r2 - r2md) < 1e-25 and all(M(r2md) <= M(r2md + e) for e in (-1e-6, 1e-6))
    # (3) 일반 n: 무작위 점 집합에서 S_n 최소 r 을 직접 구하고 md 식(D(i) 평균)과 비교
    import random
    random.seed(7)
    s3, _ = md_answer(1, "3").split("[", 1)
    i, n = sp.symbols("i n", integer=True)
    Df = sp.Function("D")
    expr = sp.sympify(s3.strip(), locals={"D": Df, "i": i, "n": n})
    ok3 = True
    for _ in range(5):
        nn = random.randint(2, 9)
        pts = [sp.Matrix([random.randint(-9, 9), random.randint(-9, 9)]) for _ in range(nn)]
        cc = sum(pts, sp.zeros(2, 1)) / nn
        dist = [sp.sqrt((p - cc).dot(p - cc)) for p in pts]
        Sn = sum((d - r) ** 2 for d in dist)
        rmin = sp.solve(sp.diff(Sn, r), r)[0]
        val = expr.subs(n, nn).doit().replace(Df, lambda z: dist[int(z) - 1])
        ok3 &= sp.simplify(sp.nsimplify(val) - rmin) == 0 or abs(sp.N(val - rmin, 30)) < 1e-25
    report(1, ok1 and ok2 and ok3, f"c={tuple(c)} r1={sp.nsimplify(r1)} r2~{mp.nstr(rg, 8)} (S_n 무작위 5세트 ok={ok3})")


def p02():
    t, t1, t2 = sp.symbols("t t1 t2", positive=True)
    v1 = [(20 * t - sp.Rational(5, 2) * t**2, 0, 6), (90 - 10 * t, 6, 9),
          (-10 + sp.Rational(10, 27) * (12 - t) ** 3, 9, 12)]
    y26 = 10 + sp.integrate(20 * t - sp.Rational(5, 2) * t**2, (t, 0, 6))
    ok1 = y26 == sp.sympify(md_answer(2, "1"))
    y112 = 0 + sum(sp.integrate(f, (t, a, b)) for f, a, b in v1)
    ok2 = y112 == sp.sympify(md_answer(2, "2"))
    T2 = sp.solve(sp.Eq(y112 + sp.integrate(-10, (t, 12, t2)), 0), t2)[0]
    y2t2 = y26 + sp.integrate(30, (t, 6, t1)) + sp.integrate(30 / (t2 - t1) * (t2 - t), (t, t1, t2))
    T1 = sp.solve(sp.Eq(sp.simplify(y2t2.subs(t2, T2)), 910), t1)
    T1 = [s for s in T1 if 6 < s < T2]
    ok3 = len(T1) == 1 and T1[0] == sp.sympify(md_answer(2, "3"))
    # 연속성 점검(v1 이 t=9 에서 0 → 최고점, t=12 에서 -10)
    okc = v1[1][0].subs(t, 9) == 0 and v1[2][0].subs(t, 12) == -10 and v1[2][0].subs(t, 9) == 0
    report(2, ok1 and ok2 and ok3 and okc, f"y2(6)={y26} y1(12)={y112} t2={T2} t1={T1}")


def p03():
    x, u = sp.symbols("x u", positive=True)
    n = sp.Symbol("n", positive=True, integer=True)
    # (1)(2) 도착시각 U~균등[0,1], 하루 P(U>1/n)=∫_{1/n}^1 du, n일 독립
    day = lambda nn: sp.integrate(1, (u, sp.Rational(1, nn) if nn != 0 else 0, 1))
    PA = lambda nn: day(nn) ** nn
    d1 = kv(md_answer(3, "1"))
    ok1 = sp.sympify(d1["P(A1)"]) == PA(1) and sp.sympify(d1["P(A2)"]) == PA(2)
    PAn = sp.sympify(md_answer(3, "2"), locals={"n": n})
    ok2 = all(sp.simplify(PAn.subs(n, k) - PA(k)) == 0 for k in range(1, 12))
    # (3)
    f = (1 - 1 / x) ** x
    d = sp.diff(sp.expand_log(sp.log(f), force=True), x)
    ok3 = all(abs(sp.N((d - sp.sympify(md_answer(3, "3"), locals={"x": x})).subs(x, v), 30)) < 1e-25
              for v in (sp.Rational(11, 10), 2, sp.pi, 50))
    # (4) [증명-수치확인] ln u < u-1 (u≠1) 과 d/dx ln f > 0 (x>1 격자 1000점), P(A_{n+1})>P(A_n) n=1..200
    import mpmath as mp
    mp.mp.dps = 40
    dl = sp.lambdify(x, d, "mpmath")
    xs = [1 + mp.mpf(k) / 50 for k in range(1, 1001)]
    ok4a = all(dl(xv) > 0 for xv in xs) and all(mp.log(xv / (xv - 1)) < 1 / (xv - 1) for xv in xs)
    seq = [mp.mpf(0)] + [(1 - mp.mpf(1) / k) ** k for k in range(2, 202)]
    ok4b = all(seq[k + 1] > seq[k] for k in range(200))
    # (5) [증명-수치확인] 극한 1/e, P(A_n) < 1/e < 1/2 → P(B_n)=1-P(A_n) > 1/2
    lim = sp.limit((1 - 1 / n) ** n, n, sp.oo)
    ok5 = lim == sp.exp(-1) and sp.exp(-1) < sp.Rational(1, 2) and all(1 - s > mp.mpf(1) / 2 for s in seq) \
        and all(s < mp.e ** -1 for s in seq)
    report(3, ok1 and ok2 and ok3 and ok4a and ok4b and ok5,
           f"P(A1),P(A2)=({PA(1)},{PA(2)}) d/dx ln f ok={ok3} [증명-수치확인] 증가 ok={ok4a and ok4b} lim={lim}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
