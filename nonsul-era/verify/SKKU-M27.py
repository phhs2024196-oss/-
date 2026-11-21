#!/usr/bin/env python3
"""SKKU-M27 (성균관대 2027학년도 모의논술 수리형) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

PREFIX = "SKKU-M27"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


# ── 문제 1: 정삼각형 개수 모델 — 기울기 ±√3 직선 k, n-k 개 → k(n-k) 개, 한 직선을 다른 방향으로 두면 a_{n-1}
def a_of(n):
    return max(k * (n - k) for k in range(n + 1))


def b_of(n):
    cand = {k * (n - k) for k in range(n + 1)} | {a_of(n - 1)}
    vals = sorted(cand, reverse=True)
    return vals[1]


def p01():
    ok0 = a_of(3) == 2 and b_of(3) == 1                       # 제시문 예시와 일치
    sa = sum(a_of(n) for n in range(3, 51))
    sb = sum(b_of(n) for n in range(3, 51))
    ok1 = sa == int(md_answer(1, "i"))
    ok2 = sb == int(md_answer(1, "ii"))
    # (iii) Π a_n / Π b_n <= 9 인 N 의 개수 (비율은 12 로 수렴하는 증가열 → N<=2000 이면 충분)
    cnt, r, goods = 0, Fraction(1), []
    for N in range(3, 2001):
        r *= Fraction(a_of(N), b_of(N))
        if r <= 9:
            cnt += 1
            goods.append(N)
    ok3 = cnt == int(md_answer(1, "iii"))
    report(1, ok0 and ok1 and ok2 and ok3, f"Σa={sa} Σb={sb} N 개수={cnt} (N={goods[0]}..{goods[-1]})")


# ── 문제 2
def F(c):
    u = c - math.floor(c)
    return -2 * u * u + u if u <= 0.5 else 2 * u * u - 3 * u + 1


def Fp(c):
    u = c - math.floor(c)
    return 1 - 4 * u if u <= 0.5 else 4 * u - 3


def D_of(c):
    """C(c,F(c)) 의 접선과 직선 AB 의 교점"""
    m = Fp(c)
    # y = m(x-c)+F(c) = -4x/3 - 1
    xx = (m * c - F(c) - 1) / (m + 4 / 3)
    return xx, -4 * xx / 3 - 1


def p02():
    A = sp.Matrix([0, -1]); B = sp.Matrix([3, -5])
    slope = (B[1] - A[1]) / (B[0] - A[0])
    ab = sp.expand(slope * x + A[1])
    n = sp.symbols("n", integer=True)
    Dx = sp.solve(sp.Eq(x - n, ab), x)[0]                     # C(n,0): F(n)=0, F'(n)=1 (제시문3)
    D = (sp.simplify(Dx), sp.simplify(Dx - n))
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "i").replace(" ", "").split(";"))
    md_ab = sp.sympify(parts["AB:y"], locals={"x": x})
    md_D = sp.sympify(parts["D"], locals={"n": n})
    ok1 = sp.simplify(md_ab - ab) == 0 and all(sp.simplify(md_D[i] - D[i]) == 0 for i in range(2))
    # 접선 기울기 범위 [-1,1] ∌ -4/3 (유일성)
    us = np.linspace(0, 1, 100001)
    fp = np.where(us <= 0.5, 1 - 4 * us, 4 * us - 3)
    ok1 = ok1 and fp.min() >= -1 - 1e-12 and fp.max() <= 1 + 1e-12
    # (ii) 2:1 외분점, 접선이 그 점을 지나는 c∈[0,2027] 개수 — 반주기별 u 의 이차식(설정식 그대로) 근 세기
    E = (2 * B - 1 * A) / (2 - 1)
    ex, ey = float(E[0]), float(E[1])
    roots = set()
    for nn in range(0, 2028):
        # 0<=u<=1/2: F=-2u^2+u, F'=1-4u ;  1/2<=u<1: F=2u^2-3u+1, F'=4u-3
        for lo, hi, Fc, Fpc in ((0.0, 0.5, (-2, 1, 0), (-4, 1)), (0.5, 1.0, (2, -3, 1), (4, -3))):
            # ey = Fp(u)*(ex-nn-u) + F(u)  → 다항식 in u
            uu = np.polynomial.Polynomial([0, 1])
            Fu = Fc[0] * uu**2 + Fc[1] * uu + Fc[2]
            Fpu = Fpc[0] * uu + Fpc[1]
            P = Fpu * (ex - nn - uu) + Fu - ey
            for r in P.roots():
                if abs(r.imag) < 1e-12:
                    u = r.real
                    if lo - 1e-12 <= u <= hi + 1e-12 and u < 1 - 1e-12:
                        c = nn + u
                        if -1e-12 <= c <= 2027 + 1e-12:
                            roots.add(round(c, 9))
    ok2 = len(roots) == int(md_answer(2, "ii")) and (ex, ey) == (6.0, -9.0)
    # (iii) 0<=c<=2027 에서 D 의 x 범위 → 선분 길이 = (5/3)·폭 (직선 AB 기울기 -4/3)
    xs = []
    ugrid = np.unique(np.concatenate([np.linspace(0, 1, 401), [0.5]]))
    for nn in range(0, 2027):
        for u in ugrid:
            xs.append(D_of(nn + u)[0])
    xs.append(D_of(2027)[0])
    L = (max(xs) - min(xs)) * 5 / 3
    ok3 = abs(L - float(sp.sympify(md_answer(2, "iii")))) < 1e-6
    # (iv) 작은 범위(C∈[0,N+1], N=1,2)에서 선분 CD 들의 합집합 넓이를 열 단위로 수치 측정 →
    #      일반식 S_N 을 독립 적분으로 만들고 비교, 그 식에 N=2026 대입한 값과 md 정답 비교
    nsym = sp.symbols("N")
    aM, aQ = -3 * nsym - sp.Rational(9, 2), 3 * nsym / 7
    top = -x + nsym + sp.Rational(1, 2)
    SN = sp.integrate(top - (-sp.Rational(4, 3) * x - 1), (x, aM, aQ)) + \
        sp.integrate(top - (x - nsym - 1), (x, aQ, nsym + sp.Rational(1, 2))) + \
        sp.integrate(2 * (x - nsym)**2 - 3 * (x - nsym) + 1 - (x - nsym - 1), (x, nsym + sp.Rational(1, 2), nsym + 1))
    SN = sp.expand(SN)
    ok4 = sp.simplify(SN.subs(nsym, 2026) - sp.sympify(md_answer(2, "iv"))) == 0
    num_ok = True
    msgs = []
    for N in (1, 2):
        cs = np.linspace(0, N + 1, 30001)
        Cx = cs; Cy = np.array([F(c) for c in cs])
        Dd = np.array([D_of(c) for c in cs]); Dx_, Dy_ = Dd[:, 0], Dd[:, 1]
        xl = min(Dx_.min(), Cx.min()); xr = max(Dx_.max(), Cx.max())
        cols = np.linspace(xl, xr, 6001)[1:-1]
        area = 0.0
        dx = cols[1] - cols[0]
        lo_x = np.minimum(Cx, Dx_); hi_x = np.maximum(Cx, Dx_)
        for X in cols:
            valid = (lo_x <= X) & (X <= hi_x) & (hi_x - lo_x > 1e-12)
            if not valid.any():
                continue
            t = (X - Dx_) / np.where(Cx != Dx_, Cx - Dx_, 1)
            Y = Dy_ + t * (Cy - Dy_)
            idx = np.nonzero(valid)[0]
            breaks = np.nonzero(np.diff(idx) > 1)[0]
            runs = np.split(idx, breaks + 1)
            iv = sorted((Y[r].min(), Y[r].max()) for r in runs)
            tot, cur_a, cur_b = 0.0, iv[0][0], iv[0][1]
            for a_, b_ in iv[1:]:
                if a_ > cur_b:
                    tot += cur_b - cur_a; cur_a, cur_b = a_, b_
                else:
                    cur_b = max(cur_b, b_)
            tot += cur_b - cur_a
            area += tot * dx
        want = float(SN.subs(nsym, N))
        num_ok = num_ok and abs(area - want) / want < 2e-3
        msgs.append(f"N={N}: union~{area:.4f} vs S_N={want:.4f}")
    report(2, ok1 and ok2 and ok3 and ok4 and num_ok,
           f"D(n)={D} roots={len(roots)} L={L:.6f} S_N={SN} | " + "; ".join(msgs))


# ── 문제 3: 정수 연산으로 구간 번호 직접 판정. 값 v 의 번호 = floor(3v) mod 6 + 1, floor(3√M)=isqrt(9M)
def label(n, k):
    return math.isqrt(9 * (n * n + k * n)) % 6 + 1


def p03():
    l500, l2027 = label(500, 21), label(2027, 21)
    parts = dict(seg.split(":") for seg in md_answer(3, "i").replace(" ", "").split(";"))
    ok1 = int(parts["a500"]) == l500 and int(parts["a2027"]) == l2027
    c2 = sum(1 for n in range(500, 2028) if label(n, 41) == 5)
    ok2 = c2 == int(md_answer(3, "ii"))

    def Acnt(k):
        return sum(1 for n in range(500, 2028) if label(n, k) == 3)
    tot = sum(Acnt(4 * m + 1) + Acnt(4 * m + 2) + Acnt(4 * m + 3) for m in range(5, 11))
    detail = [Acnt(4 * m + 2) for m in range(5, 11)]
    ok3 = tot == int(md_answer(3, "iii"))
    report(3, ok1 and ok2 and ok3, f"a500→{l500} a2027→{l2027} count5={c2} A_(4m+2)={detail} total={tot}")


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
