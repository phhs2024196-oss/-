#!/usr/bin/env python3
"""AJOU-G22A (아주대 2022 실전 자연 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G22A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(no: int, sub: str) -> dict:
    """'a=..; b=..' 꼴 answer.value 를 사전으로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

import math


def _reflect_points(theta, count):
    """I(1,0)에서 발사각 theta 로 쏜 빛의 반사지점들(수치 시뮬레이션: 원과의 교점 + 법선 반사)."""
    P = (1.0, 0.0)
    d = (-math.cos(theta), math.sin(theta))
    pts = []
    for _ in range(count):
        k = -2 * (P[0] * d[0] + P[1] * d[1])           # |P + k d| = 1 의 다른 근
        P = (P[0] + k * d[0], P[1] + k * d[1])
        pts.append(P)
        dn = d[0] * P[0] + d[1] * P[1]
        d = (d[0] - 2 * dn * P[0], d[1] - 2 * dn * P[1])  # 법선(=OP) 대칭 반사
    return pts


def p01():
    th, a = sp.symbols("theta a", positive=True)
    # (1-1) 범위: 격자에서 「2번째 제2사분면, 3번째 제3사분면」인 theta 의 최소·최대
    good = []
    for i in range(1, 20000):
        tv = i * (math.pi / 2) / 20000
        q2, q3 = _reflect_points(tv, 3)[1:]
        if q2[0] < 0 < q2[1] and q3[0] < 0 and q3[1] < 0:
            good.append(tv)
    ans = kv(1, "1-1")
    lo, hi = float(sp.sympify(ans["lo"])), float(sp.sympify(ans["hi"]))
    ok1 = abs(min(good) - lo) < 1e-3 and abs(max(good) - hi) < 1e-3
    xP = sp.sympify(ans["xP"], locals={"theta": th})
    for tv in (0.8, 0.9, 1.0):
        q2, q3 = _reflect_points(tv, 3)[1:]
        xcross = q2[0] - q2[1] * (q3[0] - q2[0]) / (q3[1] - q2[1])
        ok1 = ok1 and abs(xcross - float(xP.subs(th, tv))) < 1e-9
    # (1-2) theta=7pi/24 에서 2번째 반사지점~P 거리
    tv = 7 * math.pi / 24
    q2, q3 = _reflect_points(tv, 3)[1:]
    xcross = q2[0] - q2[1] * (q3[0] - q2[0]) / (q3[1] - q2[1])
    dist = math.hypot(q2[0] - xcross, q2[1])
    md2 = float(sp.sympify(md_answer(1, "1-2"), locals={"a": a}).subs(a, sp.cos(sp.pi / 24)))
    ok2 = abs(dist - md2) < 1e-9
    # (2) theta=59pi/128, 호 [18pi/64, 21pi/64] 와 두 번째로 만나는 반사 번호
    pts = _reflect_points(59 * math.pi / 128, 200)
    hits = [i + 1 for i, P in enumerate(pts)
            if 18 * math.pi / 64 - 1e-9 <= math.atan2(P[1], P[0]) % (2 * math.pi) <= 21 * math.pi / 64 + 1e-9]
    ok3 = hits[1] == int(md_answer(1, "2"))
    # (3) [증명-수치확인] alpha/pi 무리수, n<=10^5 에서 I 로 돌아오지 않음 + 원 해설 둘째 식 확인
    al = 1 - 2 / sp.sqrt(6)
    nn, pp, r6 = sp.symbols("nn pp r6", positive=True)
    sol = sp.solve(sp.Eq(nn * (1 - 2 / r6), 2 * pp), r6)[0]           # sqrt6 = 2n/(n-2p)
    mind = min(abs(((k * float(al)) / 2 + 0.5) % 1 - 0.5) for k in range(1, 100001))
    ok4 = al.is_rational is False and mind > 1e-7 and sp.simplify(sol - 2 * nn / (nn - 2 * pp)) == 0
    report(1, ok1 and ok2 and ok3 and ok4,
           f"theta∈({min(good):.4f},{max(good):.4f}) dist={dist:.6f} hits={hits[:2]} "
           f"[증명-수치확인] min|nα/2π-정수|={mind:.2e} (원 해설 √6=n/(n-2p) 는 2n/(n-2p) 오식)")


def p02():
    a_, s = sp.symbols("a s", positive=True)
    X = sp.symbols("X", real=True)
    # 설정: 중심 (a,0) 에서 y=x^2 위 점까지 거리 최소점 → 접점 s, 반지름 r (㉠ 사용 없이)
    D2 = (X - a_)**2 + X**4
    crit = sp.diff(D2, X)                                     # 2(X-a)+4X^3
    a_of_s = sp.solve(crit.subs(X, s), a_)[0]                 # a = 2s^3 + s
    r = sp.sqrt(D2.subs({X: s, a_: a_of_s}))
    # (2-1) sin(theta)=t/r -> 1, theta -> pi/2  [증명-수치확인]
    theta = sp.atan(s**2 / (a_of_s - s))
    lim1 = sp.limit(theta, s, 0, "+")
    ok1 = sp.simplify(lim1 - sp.pi / 2) == 0
    # (2-2)(1) a^n/r
    md21 = sp.sympify(md_answer(2, "2-1"), locals={"s": s, "n": n})
    ok2 = all(abs(float(((a_of_s**k) / r - md21.subs(n, k)).subs(s, sv))) < 1e-12
              for k in range(0, 5) for sv in (0.1, 0.7, 1.3))
    # (2-2)(2) 극한이 정수인 n (0..12 조사)
    ints = [k for k in range(0, 13) if (lambda L: L.is_finite and L.is_integer)(sp.limit(a_of_s**k / r, s, 0, "+"))]
    rng = sp.sympify(md_answer(2, "2-2"))
    ok3 = ints == [k for k in range(0, 13) if k in rng]
    # (2-3)(1) S = 2*(포물선 아래 + 선분 l1 아래)
    line = s**2 * (a_of_s - X) / (a_of_s - s)
    S = 2 * (sp.integrate(X**2, (X, 0, s)) + sp.integrate(line, (X, s, a_of_s)))
    ok4 = sp.simplify(S - sp.sympify(md_answer(2, "3-1"), locals={"s": s})) == 0
    # (2-3)(2) T = 중심각 2theta 부채꼴
    T = sp.Rational(1, 2) * r**2 * 2 * theta
    lim2 = sp.limit(T / S, s, 0, "+")
    ok5 = sp.simplify(lim2 - sp.sympify(md_answer(2, "3-2"))) == 0
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] lim θ={lim1}, 정수 극한 n={ints}, S={sp.expand(sp.simplify(S))}, lim T/S={lim2}")


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
