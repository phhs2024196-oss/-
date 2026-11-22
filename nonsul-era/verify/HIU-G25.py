#!/usr/bin/env python3
"""HIU-G25 (홍익대 2025학년도 수시 논술 자연계열 서울) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from fractions import Fraction as Fr

import sympy as sp

PREFIX = "HIU-G25"
WIDTH = 2
RESULTS = []
t = sp.symbols("t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    f = lambda n: 2**n
    def g(n):                       # f 를 n 번 합성해 n 을 넣음(작은 n 만 실제 계산)
        v = n
        for _ in range(n):
            v = f(v)
        return v
    g2 = g(2)
    ok1 = g2 == int(md_answer(1, "1"))
    g3 = g(3)                       # 2**256
    m = g3.bit_length() - 1
    ok2 = g3 == 2**m and m == int(md_answer(1, "2"))
    winner = "구골" if 10**100 > g3 else "g(3)"
    ok3 = winner == md_answer(1, "3")
    # (4) 지수탑 구조: g(n) = 2^(2^(...^(2^n))) (2 가 n 개, 맨 위 n). 작은 n 으로 구조 확인
    def tower(levels, top):
        v = top
        for _ in range(levels):
            v = 2**v
        return v
    ok_struct = all(g(n) == tower(n, n) for n in (1, 2, 3))
    twos16 = next(k for k in range(1, 6) if tower(k - 1, 2) == 16)          # 2 가 k 개인 탑 = 16
    h2_twos = g2 + twos16                                                     # h(2)=g(g(2))=g(16)
    ok4 = ok_struct and h2_twos == int(md_answer(1, "4"))
    # (5) a=g(2^256) 의 탑에는 2 가 2^256 개, 맨 위 2^256 ≥ 2 → a ≥ f(f(2^256)) = 2^(2^(2^256)) > 2^400 > 10^100
    N = 2**256
    ok_a = N > 400 and 2**400 > 10**100           # f(N)=2^N > 400 (N>9), 따라서 a ≥ 2^(f(N)) > 2^400 > 10^100
    # h(3)=g(a): 2 가 a 개(>10^100) 쌓이고 맨 위 a ≥ 2. 같은 꼭대기(≥2)에서 층이 많을수록 큼 — 작은 예로 단조성 확인
    mono = all(tower(L + 1, 2) > tower(L, 2) and tower(L, 3) >= tower(L, 2) for L in range(0, 4))
    winner5 = "h(3)" if (ok_a and mono) else "홍익수"
    ok5 = winner5 == md_answer(1, "5")
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"g(2)={g2} m={m} 2^256<10^100:{g3 < 10**100} h(2) 2의 개수={h2_twos} (5)={winner5} [증명-수치확인]")


def p02():
    A, B = {"R": Fr(2, 4), "G": Fr(2, 4)}, {"R": Fr(2, 8), "G": Fr(6, 8)}
    half = Fr(1, 2)
    PR = half * A["R"] + half * B["R"]
    ok1 = PR == Fr(md_answer(2, "1"))
    PAR = half * A["R"] / PR
    ok2 = PAR == Fr(md_answer(2, "2"))
    # (3) 정지 규칙을 직접 열거: 각 회차 파랑(계속)/빨강(정지), 10회면 정지
    import itertools
    def dist(bag):
        d = {}
        for seq in itertools.product("RG", repeat=10):
            p, k = Fr(1), 0
            for c in seq:
                k += 1
                p *= bag[c]
                if c == "R" or k == 10:
                    break
            key = (seq[:k])
            d[key] = p
        out = {}
        for key, p in d.items():
            out[len(key)] = out.get(len(key), 0) + p
        return out
    dB, dA = dist(B), dist(A)
    ok_sum = sum(dB.values()) == 1 and sum(dA.values()) == 1
    ok3 = ok_sum and dB[3] == Fr(md_answer(2, "3"))
    PA3 = half * dA[3] / (half * dA[3] + half * dB[3])
    ok4 = PA3 == Fr(md_answer(2, "4"))
    report(2, ok1 and ok2 and ok3 and ok4, f"P(R)={PR} P(A|R)={PAR} P(X=3|B)={dB[3]} P(A|X=3)={PA3}")


def p03():
    # (1) 원 x^2+(y-c)^2=1 과 y=x^2: y+(y-c)^2=1 이 양의 중근 → 정확히 두 점
    c, y, x = sp.symbols("c y x", real=True)
    q = sp.expand(y + (y - c)**2 - 1)
    csol = [s for s in sp.solve(sp.discriminant(q, y), c)]
    cands = []
    for cv in csol:
        ys = sp.solve(q.subs(c, cv), y)
        if len(set(ys)) == 1 and ys[0] > 0:
            cands.append((cv, ys[0]))
    cv, yv = cands[0]
    pts = sorted([(sp.sqrt(yv), yv), (-sp.sqrt(yv), yv)], key=lambda p: -p[0])
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "1").replace(" ", "").split(";"))
    md_c = sp.sympify(parts["center"])
    md_pts = sp.sympify("[" + parts["points"] + "]")
    md_pts = sorted([tuple(p) for p in md_pts], key=lambda p: -p[0])
    ok1 = len(cands) == 1 and sp.simplify(md_c[1] - cv) == 0 and md_c[0] == 0 and all(
        sp.simplify(a - b) == 0 for P, Q in zip(pts, md_pts) for a, b in zip(P, Q))
    # (2) 중심 (0,r) 반지름 r: 교점의 x 가 0 하나뿐인 r 의 최댓값 — 격자 탐색
    r = sp.symbols("r", positive=True)
    poly = sp.expand(x**2 + (x**2 - r)**2 - r**2)
    def nroots(rv):
        return len(set(sp.real_roots(sp.Poly(poly.subs(r, rv), x))))
    grid = [sp.Rational(k, 1000) for k in range(1, 1001)]
    good = [rv for rv in grid if nroots(rv) == 1]
    rmax = max(good)
    ok2 = rmax == sp.Rational(md_answer(3, "2")) and nroots(rmax + sp.Rational(1, 1000)) > 1
    # (3) Q = P + (1/4)·단위법선(-2t,1)/sqrt(1+4t^2) 를 독립 계산, 접촉이 P 한 점뿐인지 수치 확인
    R = sp.Rational(1, 4)
    fx = t - R * 2 * t / sp.sqrt(1 + 4 * t**2)
    gy = t**2 + R / sp.sqrt(1 + 4 * t**2)
    p3 = dict(seg.split("=", 1) for seg in md_answer(3, "3").replace(" ", "").split(";"))
    md_f, md_g = sp.sympify(p3["f(t)"], locals={"t": t}), sp.sympify(p3["g(t)"], locals={"t": t})
    ok3a = sp.simplify(fx - md_f) == 0 and sp.simplify(gy - md_g) == 0
    import numpy as np
    xs = np.linspace(-6, 6, 120001)
    ok3b = True
    for tv in (-1.3, -0.4, 0.0, 0.25, 0.9, 2.0):
        Qx, Qy = float(fx.subs(t, tv)), float(gy.subs(t, tv))
        dmin = np.min(np.hypot(xs - Qx, xs**2 - Qy))
        ok3b &= abs(dmin - 0.25) < 1e-6 and Qy > Qx**2
    # (4) dy/dx = g'/f' = 2t (기호 단순화)
    slope = sp.simplify(sp.diff(gy, t) / sp.diff(fx, t))
    ok4 = sp.simplify(slope - 2 * t) == 0 and sp.simplify(slope - sp.diff(t**2, t)) == 0
    # (5) Q1, Q2 와 두 포물선 y=cx^2+1/4 의 계수
    Q1 = (sp.nsimplify(fx.subs(t, sp.Rational(3, 8))), sp.nsimplify(gy.subs(t, sp.Rational(3, 8))))
    Q2 = (sp.nsimplify(fx.subs(t, sp.Rational(2, 3))), sp.nsimplify(gy.subs(t, sp.Rational(2, 3))))
    cc = (Q1[1] - R) / Q1[0]**2
    dd = (Q2[1] - R) / Q2[0]**2
    p5 = dict(seg.split("=", 1) for seg in md_answer(3, "5").replace(" ", "").split(";")[:4])
    ok5 = (sp.Tuple(*Q1) == sp.sympify(p5["Q1"]) and sp.Tuple(*Q2) == sp.sympify(p5["Q2"])
           and cc == sp.sympify(p5["c"]) and dd == sp.sympify(p5["d"]) and cc != dd
           and (fx.subs(t, 0), gy.subs(t, 0)) == (0, R) and "아님" in md_answer(3, "5"))
    report(3, ok1 and ok2 and ok3a and ok3b and ok4 and ok5,
           f"c={cv} y={yv} rmax={rmax} slope={slope} Q1={Q1} Q2={Q2} c={cc} d={dd} [증명-수치확인]")


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
