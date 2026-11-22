#!/usr/bin/env python3
"""EWHA-M26B (이화여대 2026학년도 모의논술 자연계열Ⅱ) — sympy 검산. 정답 하드코딩 비교 금지."""
import re
import sys
import random
import pathlib

import sympy as sp
import yaml

WIDTH = 2
RESULTS = []


def item(no):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8"))


def md_answer(no, sub):
    return next(s["answer"]["value"] for s in item(no)["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

PREFIX = "EWHA-M26B"
import mpmath as mp
from fractions import Fraction as F
from math import gcd
mp.mp.dps = 30


def p01():
    x = sp.symbols("x", real=True)
    s, c = sp.sin(x), sp.cos(x)
    ok1 = sp.simplify(sp.diff(s**2 - c**2, x) - 4*s*c) == 0
    ok1 &= sp.simplify(s**2*c**2 - sp.Rational(1, 2)*(1 - s**4 - c**4)) == 0
    S = lambda n: mp.quad(lambda t: mp.sin(t)*mp.cos(t)/(2*mp.sin(t)**4+2*mp.cos(t)**4)**n, [0, mp.pi/4, mp.pi/2])
    Sv = {n: S(n) for n in range(1, 16)}
    ok2 = all(abs(Sv[n+1] - ((2*n-1)*Sv[n]/(2*n) + mp.mpf(1)/(2**(n+2)*n))) < mp.mpf(10)**-20 for n in range(1, 15))
    A = [None, F(2)]
    for m in range(2, 2001):
        A.append(A[-1]*F(2*m, 2*m-1))
    ok3 = all(2*m+1 <= A[m]**2 <= 4*m for m in range(1, 2001))
    ok3 &= all((2*k+1)*(2*k+3) < (2*k+2)**2 and 4*k*(k+1) < (2*k+1)**2 for k in range(1, 2001))
    # (4) 망원 항등식 + 극한(α<1/2): 정확 항등식과 상계 감소 확인
    Af = lambda k: mp.mpf(A[k].numerator)/A[k].denominator
    ok4 = all(abs(Af(k)*Sv[k+1] - Af(k-1)*Sv[k] - Af(k)/(2**(k+2)*k)) < mp.mpf(10)**-20 for k in range(2, 15))
    for al in (mp.mpf("0.49"), mp.mpf("0.3"), mp.mpf(-1)):
        def T(n):
            return sum(n**al*mp.sqrt(k)/Af(n)*Af(k)/(2**(k+2)*k) for k in range(2, n+1))
        vals = [T(n) for n in (100, 400, 1600)]
        ok4 &= all(v >= 0 for v in vals) and vals[0] > vals[1] > vals[2]
        ok4 &= all(T(n) <= n**al/mp.sqrt(2*n+1)*(mp.mpf(1)/4 - mp.mpf(1)/2**(n+1)) + mp.mpf(10)**-25 for n in (10, 100, 1000))
    report(1, ok1 and ok2 and ok3 and ok4, "[증명-수치확인] (1) 기호항등식 (2) S_n 수치적분 n<=15 점화식 (3) m<=2000 정확계산 (4) 망원식·상계·감소")


def f(x):
    x = abs(x)
    if x < 1:
        return abs(mp.cos(mp.pi*x))
    n = int(mp.floor(x))
    u = 2**n*(x-n)
    return 1 + (mp.mpf(-1)/3)**n*(f(u - mp.floor(u)) - 1)


def p02():
    ok1 = all(abs(f(mp.mpf(n)) - 1) < 1e-25 for n in range(1, 12))
    # (2) 구간 [n,n+1)의 각 조각에서 |cos(pi u)| = c_n 의 해 개수 (u in [0,1))
    def nsol(c):
        if c == 1: return 1
        if 0 < c < 1: return 2
        if c == 0: return 1
        return 0
    target = sp.Rational(8, 9)
    cnt = nsol(target)                     # [0,1)
    for n in range(1, 40):                 # n>=3 에서 |c_n|>1 로 0개
        cn = 1 + (target - 1)/sp.Rational(-1, 3)**n
        cnt += 2**n*nsol(cn)
    if f(0) == target:
        cnt = 2*cnt - 1
    else:
        cnt = 2*cnt
    # 수치 교차 확인: 2<=x<3 해 x=17/8.. 에서 f=8/9, 그 외 격자에서 f>8/9
    ok2 = cnt == int(md_answer(2, "2"))
    ok2 &= all(abs(f(mp.mpf(k)/8) - mp.mpf(8)/9) < 1e-25 for k in (17, 19, 21, 23))
    ok2 &= min(f(mp.mpf(3) + mp.mpf(i)/997) for i in range(0, 4000)) > mp.mpf(8)/9
    # (3) 조각별 고정밀 적분
    pts = [0, 0.5, 1, 1.25, 1.5, 1.75, 2] + [2 + j/8 for j in range(1, 9)]
    I = 2*mp.quad(f, pts)
    ok3 = abs(I - mp.mpf(sp.N(sp.sympify(md_answer(2, "3")), 40))) < mp.mpf(10)**-15
    report(2, ok1 and ok2 and ok3, f"f(n)=1 n<=11, 교점 {cnt}개(구간별 해 개수 독립계산), 적분 {mp.nstr(I, 15)}")


def simulate(m, maxpts=200):
    """반사 규칙을 그대로 시뮬레이션(유리수 정확). 반환: 점 목록."""
    P = [(F(0), F(0)), (F(1), m)]
    slope = -m
    verts = {(F(0), F(0)), (F(1), F(0)), (F(1), F(1)), (F(0), F(1))}
    if P[1] in verts:
        return P
    while len(P) < maxpts:
        x0, y0 = P[-1]
        cands = []
        for xe in (F(0), F(1)):
            y = y0 + slope*(xe - x0)
            if xe != x0 and 0 <= y <= 1:
                cands.append((xe, y))
        if slope != 0:
            for ye in (F(0), F(1)):
                xx = x0 + (ye - y0)/slope
                if ye != y0 and 0 <= xx <= 1:
                    cands.append((xx, ye))
        cands = [c for c in set(cands) if c != (x0, y0) and c != P[-2]]
        # 진행 방향: 직전 선분과 반사되는 쪽 — 후보 중 직전 점과 다른 하나
        nxt = cands[0]
        P.append(nxt)
        if nxt in verts:
            return P
        # 다음 반사: 세로 변이면 기울기 부호 반전, 가로 변이면 부호 반전 (어느 쪽이든 기울기 부호만 바뀜)
        slope = -slope
    return P


def p03():
    pts = simulate(F(2, 3))
    md1 = [tuple(F(str(sp.Rational(sp.sympify(v)))) for v in p) for p in sp.sympify(md_answer(3, "1"))]
    ok1 = pts == md1
    # (2) 끝점이 C 인 유리 기울기만 가능(무리수면 꼭짓점 도달 불가). 분모<=60 전수
    sel = set()
    for q in range(1, 61):
        for p in range(1, q+1):
            mm = F(p, q)
            if not (F(1, 20) < mm < 1):
                continue
            P = simulate(mm, 400)
            if P[-1] == (F(0), F(1)) and not any(y == 1 and x != 0 for (x, y) in P):
                sel.add(mm)
    s2 = sum(1/mm for mm in sel)
    ok2 = s2 == int(md_answer(3, "2"))
    # (3) P15 = C : 점 개수 = a+b 이므로 분모<=30 이면 충분, 0<=m<=1 전수
    sel3 = set()
    for q in range(1, 31):
        for p in range(0, q+1):
            P = simulate(F(p, q), 50)
            if len(P) == 15 and P[-1] == (F(0), F(1)):
                sel3.add(F(p, q))
    md3 = {F(str(sp.Rational(sp.sympify(v)))) for v in sp.sympify(md_answer(3, "3"))}
    ok3 = sel3 == md3
    report(3, ok1 and ok2 and ok3, f"반사 시뮬레이션: (1) {len(pts)}점 일치, (2) 1/m 합 {s2}, (3) {sorted(sel3)}")


if __name__ == "__main__":
    fns = sorted((n for n in dir() if re.fullmatch(r"p\d+", n)), key=lambda s: int(s[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)

