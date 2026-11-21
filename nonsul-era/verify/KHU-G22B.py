#!/usr/bin/env python3
"""KHU-G22B (경희대 2022학년도 논술 자연계 11.21 오전, KHU-G22B-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "KHU-G22B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def chain_sum(vertex_angle, r0=1.0, terms=4000):
    """꼭짓점 각이 vertex_angle 인 두 변에 접하며 이어지는 원들의 반지름 합(첫 원 r0 제외).
    꼭짓점에서 중심까지 거리 d=r/sin(φ/2), 이웃 원은 외접: d_k - d_{k+1} = r_k + r_{k+1}."""
    s = math.sin(vertex_angle / 2)
    r, tot = r0, 0.0
    for _ in range(terms):
        # (r - r')/s = r + r'  →  r' = r(1-s)/(1+s)
        r = r * (1 - s) / (1 + s)
        tot += r
        if r < 1e-18:
            break
    return tot


def p01():
    """논제 I 원들의 둘레 합 f(θ), 세 변 합 g(θ), 극한"""
    th = sp.symbols("theta", positive=True)
    f_md = sp.pi * sp.sec(th) + 2 * sp.pi * sp.csc(th / 2) - sp.pi        # md 답
    g_md = 2 * sp.tan(th) + 4 * sp.cot(th / 2)
    ok = True
    for tv in (0.3, 0.7, 1.0, 1.3, 1.5):
        A = math.pi - 2 * tv
        total = 2 * math.pi * (1 + chain_sum(A) + 2 * chain_sum(tv))
        ok = ok and abs(total - float(f_md.subs(th, tv))) < 1e-9
        # 내접원 반지름 1 인 이등변삼각형의 둘레: 꼭짓점→접점 길이 = cot(각/2)
        per = 2 / math.tan(A / 2) + 4 / math.tan(tv / 2)
        ok = ok and abs(per - float(g_md.subs(th, tv))) < 1e-9
    L0 = sp.limit(f_md / g_md, th, 0, "+")
    L1 = sp.limit(f_md / g_md, th, sp.pi / 2, "-")
    ok = ok and L0 == sp.pi / 2 and L1 == sp.pi / 2
    report(1, ok, f"(1) 원 사슬 수치합 = f(θ) (5점)  (2) g 수치 일치, lim0={L0}, limπ/2={L1}")


def amin(rel):
    """위치차 R(t)=∫_0^t (v1-v2); P,Q 가 (0,1)에서 만남 ⇔ q0 ∈ R((0,1)). 최소 a = sup_{(0,1)} R (sup≥0일 때)"""
    u = sp.symbols("u")
    R = sp.integrate(rel.subs(t, u), (u, 0, t))
    crit = list(sp.solveset(sp.diff(R, t), t, sp.Interval.open(0, 1)))
    cands = [R.subs(t, c) for c in crit] + [sp.limit(R, t, 0), sp.limit(R, t, 1)]
    sup = max(cands, key=lambda v: float(v))
    # 격자 확인
    grid = max(float(R.subs(t, k / 4000)) for k in range(1, 4000))
    return sp.simplify(sup), grid, R


def p02():
    """논제 II 두 점이 만나지 않을 a 의 최솟값"""
    s1, g1, _ = amin(sp.cos(sp.pi * t / 2))
    s2, g2, R2 = amin(t * sp.cos(4 * sp.pi * t))
    ok1 = sp.simplify(s1 - 2 / sp.pi) == 0 and abs(g1 - float(s1)) < 1e-6
    ok2 = sp.simplify(s2 - (5 * sp.pi - 2) / (32 * sp.pi**2)) == 0 and abs(g2 - float(s2)) < 1e-6
    # 최솟값이 실제로 내부(t=5/8)에서 달성됨 → q0=a 는 만남이지만 조건은 q0>a 이므로 무관
    report(2, ok1 and ok2, f"(1) a_min={s1}  (2) a_min={s2}≈{float(s2):.6f}")


def p03():
    """논제 III (1) 가위바위보 확률 (2) 띠 모양 수"""
    fav = 0
    for seq in itertools.product((1, 0, -1), repeat=4):      # 1:A승 0:무 -1:B승
        wa, wb = seq.count(1), seq.count(-1)
        fav += wa >= wb          # 조기 종료해도 결정된 승자는 4회 결과와 같음
    P = sp.Rational(fav, 81)
    # 조기 종료 규칙대로 실제 진행 확률도 계산(검증)
    def prob(k, wa, wb):
        rem = 4 - k
        if wa - wb >= rem:      # B 가 남은 판을 모두 이겨도 동률 이하 → A 확정
            return sp.Integer(1)
        if wb - wa > rem:
            return sp.Integer(0)
        if rem == 0:
            return sp.Integer(1 if wa >= wb else 0)
        return sum(sp.Rational(1, 3) * prob(k + 1, wa + (o == 1), wb + (o == -1)) for o in (1, 0, -1))
    P2 = prob(0, 0, 0)
    ok1 = P == sp.Rational(50, 81) and P2 == P

    def a_n(N):
        seen = set()
        for w in itertools.product("BW", repeat=N):
            s = "".join(w)
            comp = s.translate(str.maketrans("BW", "WB"))
            reps = [x[i:] + x[:i] for x in (s, comp) for i in range(N)]
            seen.add(min(reps))
        return len(seen)
    vals = [a_n(k) for k in (2, 4, 5, 6)]
    ok2 = vals == [2, 4, 4, 8]
    report(3, ok1 and ok2, f"(1) P={P} (조기종료 재귀={P2})  (2) a2,a4,a5,a6={vals}")


def p04():
    """논제 IV 타원 접선과 삼각형 ABC"""
    a, p, q = sp.symbols("a p q", positive=True)
    # 접선 p x/a^2 + q y = 1
    yA = sp.solve(sp.Eq(p * a / a**2 + q * sp.Symbol("y"), 1), sp.Symbol("y"))[0]
    xB = sp.solve(sp.Eq(p * x / a**2 + q * 1, 1), x)[0]
    S = sp.simplify((1 - yA) * (a - xB) / 2)
    ok1 = sp.simplify(S - (p + a * q - a)**2 / (2 * p * q)) == 0
    # (2) q = sqrt(1-p^2/a^2) 대입 후 최대
    Sp = S.subs(q, sp.sqrt(1 - p**2 / a**2))
    ok2 = True
    for av in (sp.Rational(1, 2), 1, 2, 5):
        Sv = Sp.subs(a, av)
        crit = [c for c in sp.solve(sp.diff(Sv, p), p) if c.is_real and 0 < c < av]
        best = max([Sv.subs(p, c) for c in crit], key=float)
        grid = max(float(Sv.subs(p, av * k / 2000)) for k in range(1, 2000))
        ok2 = ok2 and abs(float(best) - grid) < 1e-6 and sp.simplify(best - (3 - 2 * sp.sqrt(2)) * av) == 0
    report(4, ok1 and ok2, f"(1) S={sp.factor(S)}  (2) 최댓값=(3-2√2)a (a=1/2,1,2,5 확인)")


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
