#!/usr/bin/env python3
"""HIU-G21B (홍익대 2021학년도 수시 논술 자연계열 서울 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G21B"
WIDTH = 2
RESULTS = []
x, n, N = sp.symbols("x n N")


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    out = {}
    for seg in s.split(";"):
        k, _, v = seg.partition("=")
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 초기하 확률을 정수 조합으로 직접 계산
    from math import comb
    from fractions import Fraction
    f = lambda M: Fraction(comb(20, 3) * comb(M - 20, 22), comb(M, 25))
    Ms = range(42, 400)
    ans1 = sp.sympify(md_answer(1, "1"), locals={"N": N})
    ok1 = all(sp.Rational(f(M).numerator, f(M).denominator) == sp.nsimplify(ans1.subs(N, M)) for M in (42, 50, 166, 300))
    alt = sp.sympify(next(iter([
        __import__("yaml").safe_load(open(__import__("pathlib").Path(__file__).resolve().parents[1]
                                          / "items/HIU/PAST/HIU-G21B-01.yaml", encoding="utf-8"))
        ["subparts"][0]["answer"]["equivalents"][0]])), locals={"N": N})
    ok1 = ok1 and all(sp.simplify(alt.subs(N, M) - ans1.subs(N, M)) == 0 for M in (42, 77, 166))
    g = sp.sympify(md_answer(1, "2"), locals={"N": N})
    ok2 = all(sp.Rational((f(M + 1) / f(M)).numerator, (f(M + 1) / f(M)).denominator) == g.subs(N, M) for M in Ms)
    up = [M for M in Ms if f(M + 1) > f(M)]
    down = [M for M in Ms if f(M + 1) < f(M)]
    a3 = md_answer(1, "3")
    lo, hi = map(int, re.search(r"(\d+)<=N<=(\d+)", a3).groups())
    ge = int(re.search(r"N>=(\d+)", a3).group(1))
    ok3 = up == list(range(lo, hi + 1)) and down == list(range(ge, 400))
    argmax = max(Ms, key=f)
    ok4 = argmax == int(md_answer(1, "4"))
    report(1, ok1 and ok2 and ok3 and ok4, f"f ok={ok1} g ok={ok2} up=[{up[0]},{up[-1]}] down from {down[0]} argmax={argmax}")


def pos_roots(poly):
    return len([r for r in sp.Poly(poly, x).real_roots() if r > 0])


def p02():
    import random
    rnd = random.Random(2021)
    # (1)~(4): 조건 (a)(b) 를 만족하는 무작위 다항식에서 홀짝·부등식 확인 (증명의 핵심 명제 표본 검사)
    cnt = 0
    ok = True
    while cnt < 300:
        deg = rnd.randint(2, 7)
        cs = [rnd.choice([-1, 1]) * rnd.randint(1, 30) for _ in range(deg)] + [rnd.randint(1, 5)]   # a_0..a_n
        f = sum(c * x**i for i, c in enumerate(cs))
        f1, f2 = sp.diff(f, x), sp.diff(f, x, 2)
        if any(sp.discriminant(sp.Poly(h, x)) == 0 for h in (f, f1) if sp.degree(h, x) >= 2):
            continue
        if sp.degree(f2, x) >= 2 and sp.discriminant(sp.Poly(f2, x)) == 0:
            continue
        k0, k1, k2 = pos_roots(f), pos_roots(f1), pos_roots(f2)
        a0, a1, a2 = cs[0], cs[1], cs[2] if deg >= 2 else None
        ok &= (k0 % 2 == 0) == (a0 > 0)                         # (1)
        ok &= k0 <= k1 + 1                                      # (2)
        if a0 * a1 > 0:
            ok &= k0 <= k1                                      # (3)
        if a2 is not None and a0 * a1 > 0 and a1 * a2 > 0:
            ok &= k0 <= k2                                      # (4)
        cnt += 1
    # (5) 주어진 다항식: 조건 확인 + 양의 실근 개수 직접 계산
    f = x**4 + 2718 * x**3 - 2818 * x**2 - 3141 * x - 5926
    cond = all(sp.discriminant(sp.Poly(h, x)) != 0 for h in (f, sp.diff(f, x), sp.diff(f, x, 2)))
    k0 = pos_roots(f)
    ok5 = cond and k0 == int(md_answer(2, "5"))
    report(2, ok and ok5, f"random samples={cnt} props ok={ok}; (5) conditions={cond} k0={k0}")


def p03():
    import math
    i_, j_ = sp.symbols("i j")
    chord = sp.sympify(md_answer(3, "1"), locals={"i": i_, "j": j_, "n": n})
    ok1 = True
    for m in range(3, 12):
        P = [(math.cos(2 * math.pi * k / m), math.sin(2 * math.pi * k / m)) for k in range(m)]
        for a in range(m):
            for b in range(a + 1, m):
                d = math.dist(P[a], P[b])
                ok1 &= abs(d - float(chord.subs({i_: a + 1, j_: b + 1, n: m}))) < 1e-12

    def M(m):
        P = [(math.cos(2 * math.pi * k / m), math.sin(2 * math.pi * k / m)) for k in range(m)]
        ds = [math.dist(P[a], P[b]) for a in range(m) for b in range(a + 1, m)]
        return sum(ds) / len(ds), P
    ok2 = True
    for m in range(3, 15):
        Mm, P = M(m)
        L = sum(math.dist(P[0], P[k]) for k in range(1, m)) / (m - 1)
        ok2 &= abs(Mm - L) < 1e-12
    # (3) 극한: 정적분 계산 + 큰 n 수치
    lim = sp.integrate(2 * sp.sin(sp.pi * x), (x, 0, 1))
    ans3 = sp.sympify(md_answer(3, "3"))
    ok3 = sp.simplify(lim - ans3) == 0 and abs(M(3000)[0] - float(ans3)) < 1e-3
    # (4) 전수 계수
    a4 = parts(md_answer(3, "4"))
    ok4 = True
    for m in range(2, 12):
        same = sum(1 for a in range(2 * m) for b in range(a + 1, 2 * m) if a % 2 == b % 2)
        diff = sum(1 for a in range(2 * m) for b in range(a + 1, 2 * m) if a % 2 != b % 2)
        ok4 &= same == sp.sympify(a4["same"], locals={"n": n}).subs(n, m) and diff == sp.sympify(a4["different"], locals={"n": n}).subs(n, m)
    # (5) 큰 n 에서 F_n, G_n 직접 계산
    a5 = parts(md_answer(3, "5"))
    m = 1500
    P = [(math.cos(math.pi * k / m), math.sin(math.pi * k / m)) for k in range(2 * m)]
    # 회전 대칭: 꼭짓점 0, 1 에서의 평균으로 충분(같은 색/다른 색 평균은 꼭짓점에 무관)
    same = [math.dist(P[0], P[k]) for k in range(2, 2 * m, 2)]
    diff = [math.dist(P[0], P[k]) for k in range(1, 2 * m, 2)]
    Fm, Gm = sum(same) / len(same), sum(diff) / len(diff)
    ok5 = abs(Fm - float(sp.sympify(a5["lim F_n"]))) < 2e-3 and abs(Gm - float(sp.sympify(a5["lim G_n"]))) < 2e-3
    report(3, ok1 and ok2 and ok3 and ok4 and ok5,
           f"chord={ok1} M=L={ok2} lim={lim} counts={ok4} F_{m}={Fm:.6f} G_{m}={Gm:.6f}")


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
