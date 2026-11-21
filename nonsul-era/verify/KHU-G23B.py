#!/usr/bin/env python3
"""경희대 2023학년도 논술(자연계, 11.20 오전) KHU-G23B-01~03 — sympy 검산. 정답 하드코딩 비교 금지, 문제 설정에서 독립 재계산."""
import itertools
import math
import re
import sys
from fractions import Fraction as Fr

import sympy as sp

PREFIX = "KHU-G23B"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I: y=x^2, A(t,t^2), l1 ⊥ l2 접선, C=교점."""
    t = sp.symbols("t", positive=True)
    x, u = sp.symbols("x u", real=True)
    l1 = 2*t*x - t**2
    ub = sp.solve(sp.Eq(2*u*2*t, -1), u)[0]          # 기울기 곱 -1
    l2 = 2*ub*x - ub**2
    cx = sp.solve(sp.Eq(l1, l2), x)[0]
    cy = sp.simplify(l1.subs(x, cx))
    s = sp.integrate(sp.sqrt(sp.diff(cx, t)**2 + sp.diff(cy, t)**2), (t, 1, 2))
    s = sp.nsimplify(sp.simplify(s))
    ok = s == sp.Rational(9, 16) and sp.simplify(cy + sp.Rational(1, 4)) == 0
    # (2) 넓이: 좌표 공식(신발끈)으로 독립 계산
    A = (t, t**2); B = (ub, ub**2); C = (cx, cy); P = (t, 0); O = (0, 0)
    area = lambda p, q, r: sp.Abs((q[0]-p[0])*(r[1]-p[1]) - (r[0]-p[0])*(q[1]-p[1]))/2
    St = sp.simplify(area(A, B, C)/area(A, O, P))
    L = sp.limit(St, t, sp.oo)
    ok &= L == sp.Rational(1, 2)
    report(1, ok, f"(1) C=({sp.simplify(cx)}, {cy}), s={s}; (2) lim S(t)={L}")


def p02():
    """논제 II: f=-x+1/x^2, g=-x^2+k+1/x."""
    x, k = sp.symbols("x k", positive=True)
    f = -x + 1/x**2
    g = -x**2 + k + 1/x
    ok = True
    # (1) k>0 이면 x>0 에서 교점 2개 (k 여러 값 수치)
    for kv in [sp.Rational(1, 100), sp.Rational(1, 2), 1, 3, 10, 50]:
        h = sp.together((f - g).subs(k, kv))
        num = sp.numer(h)
        roots = [r for r in sp.Poly(num, x).nroots(n=30) if abs(sp.im(r)) < 1e-20 and sp.re(r) > 0]
        ok &= len(roots) == 2
    # y=x+1/x 증감: 최소 2 at x=1
    y = x + 1/x
    ok &= sp.solve(sp.diff(y, x), x) == [1] and y.subs(x, 1) == 2
    # (2) (α-β)^2 를 k 로 — 수치 근으로 확인 후 닫힌식 유도
    T = sp.symbols("T")
    t0 = max(sp.solve(T**2 - T - 2 - k, T), key=lambda r: r.subs(k, 1))
    Sk = sp.expand(t0**2 - 4)
    for kv in [sp.Rational(1, 2), 2, 7]:
        num = sp.numer(sp.together((f - g).subs(k, kv)))
        rr = sorted(sp.re(r) for r in sp.Poly(num, x).nroots(n=30) if abs(sp.im(r)) < 1e-20 and sp.re(r) > 0)
        ok &= abs((rr[1] - rr[0])**2 - Sk.subs(k, kv)) < 1e-20
    V = sp.simplify(sp.integrate(6*Sk.subs(k, x), (x, 4, 10)))
    ok &= V == 307
    report(2, ok, f"(1) k>0 에서 양의 교점 2개(6개 k); (2) S(k)={sp.simplify(Sk)}, V={V}")


def simulate(bits, n):
    """bits[i]=1: 방 i+1 에 처음 도착한 학생이 운동장 출구 선택. 학생 1..n 진행, 상품 여부 목록 반환.
    (다) 규칙: 앞 학생이 방 i 에서 나왔으면 다음 학생은 연결된 출구로 방 (i+1)까지 그대로 간다
    → 이미 선택이 정해진 방은 통과하고, 처음 도착한 방에서만 선택한다."""
    decided = set()
    prize = []
    for _student in range(n):
        room = 1
        while True:
            if room == n + 1:
                prize.append(True)
                break
            if room not in decided:
                decided.add(room)
                if bits[room-1] == 1:
                    prize.append(False)
                    break
            room += 1
    return prize


def p03():
    """논제 III: 방 통과 게임."""
    n = 7
    PC = PCD = Fr(0)
    for bits in itertools.product((0, 1), repeat=n):
        pr = Fr(1, 2**n)
        pz = simulate(bits, n)
        if pz[5]:
            PC += pr
            if not pz[2]:
                PCD += pr
    cond = PCD/PC
    ok = cond == Fr(91, 120)
    # 낙첨자 수 = 운동장 출구 선택 수 ~ B(n,1/2) 확인 (n=7 전수)
    ok &= all(sum(1 for z in simulate(b, n) if not z) == sum(b) for b in itertools.product((0, 1), repeat=n))
    # (2) 정규근사(경희대 공식 풀이 방식, 연속성 수정 없음)
    Phi = lambda z: 0.5*(1 + math.erf(z/math.sqrt(2)))
    PA = Phi((210 - 200)/10)
    kmax = max(kk for kk in range(1, 73) if Phi(((72 - kk) - 36)/math.sqrt(18)) >= PA - 1e-15)
    ok &= kmax == 31
    # 참고: 이항분포 정확값
    def cdf(nn, m):
        return sum(math.comb(nn, i) for i in range(0, m+1))/2**nn
    PAx = cdf(400, 210)
    kex = max(kk for kk in range(1, 73) if cdf(72, 72 - kk) >= PAx)
    report(3, ok, f"(1) 2^7 전수 시뮬레이션 P(D|C)={cond}; (2) 정규근사 k최대={kmax} (참고: 이항 정확계산 k최대={kex})")


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
