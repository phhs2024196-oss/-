#!/usr/bin/env python3
"""숭실대 2024학년도 모의논술(자연) 문제 1~4 (SSU-M24-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import math
import re
import sys

import mpmath as mp
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

PREFIX = "SSU-M24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    """yaml 의 subparts[sub].answer.value 원문(문자열)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return str(d["subparts"][sub]["answer"]["value"])


def tex(s):
    """간단한 LaTeX 식 → sympy (\\dfrac, \\sqrt, \\pi, ^, 암묵적 곱)."""
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\dfrac", "\\frac")
    for _ in range(10):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", " pi ").replace("^", "**").replace("{", "(").replace("}", ")")
    return parse_expr(s, local_dict={"x": x, "t": t, "pi": sp.pi, "sqrt": sp.sqrt},
                      transformations=standard_transformations + (implicit_multiplication_application,))


def golden(fun, lo, hi, it=200, maximize=False):
    """단봉 함수의 최소(또는 최대) 위치 — 황금분할 탐색(mpmath)."""
    g = (lambda u: -fun(u)) if maximize else fun
    r = (mp.sqrt(5) - 1) / 2
    a, b = mp.mpf(lo), mp.mpf(hi)
    c, d = b - r * (b - a), a + r * (b - a)
    for _ in range(it):
        if g(c) < g(d):
            b, d = d, c
            c = b - r * (b - a)
        else:
            a, c = c, d
            d = a + r * (b - a)
    return (a + b) / 2


def clip(poly, A, B, C):
    """다각형을 반평면 A x + B y <= C 로 자른다(Sutherland–Hodgman)."""
    out = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = A * P[0] + B * P[1] - C, A * Q[0] + B * Q[1] - C
        if fp <= 0:
            out.append(P)
        if fp * fq < 0:
            s = fp / (fp - fq)
            out.append((P[0] + s * (Q[0] - P[0]), P[1] + s * (Q[1] - P[1])))
    return out


def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def p01():
    """접점 (cos u, sin u) (u∈(π,3π/2)) 의 접선 m 으로 삼각형(ℓ, x=1, y=-1)을 잘라 넓이를 직접 계산·최소화."""
    s3 = mp.sqrt(3)
    # ℓ: 기울기 √3, y절편 c>0, 원점과의 거리 1 → c/2 = 1
    c_l = mp.findroot(lambda c: c / mp.sqrt(1 + 3) - 1, 1)
    H = ((-1 - c_l) / s3, mp.mpf(-1))
    tri = [H, (mp.mpf(1), mp.mpf(-1)), (mp.mpf(1), s3 + c_l)]
    S = lambda u: area(clip(tri, mp.cos(u), mp.sin(u), 1))   # m: x cos u + y sin u = 1, 원점 쪽 영역
    grid = [S(mp.pi + (k + 0.5) * mp.pi / 2 / 400) for k in range(400)]
    kmin = min(range(400), key=lambda k: grid[k])
    u0 = golden(S, mp.pi + kmin * mp.pi / 800, mp.pi + (kmin + 1) * mp.pi / 800)
    slope, icpt = -mp.cos(u0) / mp.sin(u0), 1 / mp.sin(u0)
    ans = load_answer(1)
    rhs = tex(ans.split("=")[1])
    sl_md, ic_md = sp.diff(rhs, x), rhs.subs(x, 0)
    ok = abs(slope - mp.mpf(str(sp.N(sl_md, 30)))) < 1e-10 and abs(icpt - mp.mpf(str(sp.N(ic_md, 30)))) < 1e-10
    report(1, ok, f"최소 넓이 {mp.nstr(S(u0), 12)}, m: 기울기 {mp.nstr(slope, 12)}, y절편 {mp.nstr(icpt, 12)}, md={ans}")


def p02():
    """b=qπ (q: 1/400 격자) 마다 양의 실근 개수를 직접 세어 11개인 b 의 범위와 d 의 최댓값 확인."""
    def count(q):
        b = sp.pi * q
        # [0,1] 구간: cos(bx)=0 ⇔ bx=(2k-1)π/2, 0<x≤1
        k_cnt = sum(1 for k in range(1, 200) if sp.Rational(2 * k - 1, 2) <= q)
        c = -b * sp.sin(b)            # x=1 에서 미분가능: c = f'(1-) = -b sin b
        d = sp.cos(b) - c             # 연속: c + d = cos b
        if c == 0:
            extra = 0 if d != 0 else 10 ** 9
        else:
            xs = -d / c
            extra = 1 if sp.N(xs - 1, 50) > 0 else 0
        return k_cnt + extra, d
    qs = [sp.Rational(3600 + i, 400) for i in range(0, 1201)]   # 9π ~ 12π
    feas = [q for q in qs if count(q)[0] == 11]
    lo, hi = min(feas), max(feas)
    left_open = count(lo - sp.Rational(1, 400))[0] != 11 and count(sp.Integer(10))[0] != 11
    contig = len(feas) == int((hi - lo) * 400) + 1
    a1 = load_answer(2, 0)
    L, U = [tex(s) for s in re.split(r"<b\\le", a1)]
    ok1 = contig and left_open and sp.simplify(L - (lo - sp.Rational(1, 400)) * sp.pi) == 0 and sp.simplify(U - hi * sp.pi) == 0
    report(2, ok1, f"(2-1) 근 11개인 b/π 격자: ({lo - sp.Rational(1, 400)}, {hi}], md={a1}")
    dfun = lambda bb: bb * mp.sin(bb) + mp.cos(bb)
    dgrid = max((dfun(mp.pi * mp.mpf(str(sp.N(q, 20)))), q) for q in feas)
    bstar = golden(dfun, mp.pi * 10, mp.pi * 11, maximize=True)
    a2 = tex(load_answer(2, 1))
    ok2 = abs(dfun(bstar) - mp.mpf(str(sp.N(a2, 30)))) < 1e-20 and dgrid[0] <= dfun(bstar) + 1e-20
    report(2, ok2, f"(2-2) d 최댓값 {mp.nstr(dfun(bstar), 15)} (b/π={mp.nstr(bstar / mp.pi, 10)}), md={load_answer(2, 1)}")


def p03():
    """역함수 존재(일대일) ⇔ f'≤0 (f→+∞ at 0+, →-∞ at ∞) 를 수치로 판정해 b 의 상한을 찾고 적분 최댓값 계산."""
    bb = sp.symbols("b", real=True)
    f = -2 * x**2 + bb * x + 2 * sp.log(2) - sp.log(x)
    fp = sp.lambdify((x, bb), sp.diff(f, x), "mpmath")
    supfp = lambda b: max(fp(mp.mpf(k) / 1000, b) for k in range(1, 5001))   # x∈(0,5]
    lo, hi = mp.mpf(0), mp.mpf(10)
    for _ in range(60):                           # sup f' ≤ 0 인 b 의 상한(이분법)
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if supfp(mid) <= 1e-30 else (lo, mid)
    I = sp.integrate(f, (x, 1, 2))
    dI = sp.diff(I, bb)
    val = sp.N(I.subs(bb, sp.nsimplify(round(float(lo), 9))), 30)
    ans = tex(load_answer(3))
    ok = dI > 0 and abs(val - sp.N(ans, 30)) < 1e-8
    report(3, ok, f"b 상한≈{mp.nstr(lo, 12)}, ∫=({sp.simplify(I)}), 증가(dI/db={dI}) → 최댓값≈{sp.N(val, 12)}, md={load_answer(3)}")


def p04():
    """경우의 수 직접 계산 + 작은 경우 전수 열거로 중복 배수 확인."""
    from itertools import combinations
    C = math.comb
    A = C(30, 11) * C(11, 2)
    P, Q, R = C(30, 2) * C(28, 9), C(30, 1) * C(29, 10) * C(10, 1), C(30, 5) * C(5, 2) * C(25, 6)
    def val(tok):
        tok = tok.strip()
        m = re.fullmatch(r"\{\}_\{?(\d+)\}?\\mathrm\{([CP])\}_\{?(\d+)\}?", tok)
        if m:
            nn, kind, r = int(m.group(1)), m.group(2), int(m.group(3))
            return C(nn, r) if kind == "C" else math.perm(nn, r)
        return int(tok)
    md = [val(s) for s in re.split(r",\\ |,", load_answer(4))]
    # 전수: 학생 8명, 팀 5명, 리더 2명, 방법3 축소판(먼저 3명 → 리더 2 → 나머지 2명)
    from collections import Counter
    cnt = Counter()
    S8 = range(8)
    for first in combinations(S8, 3):
        for lead in combinations(first, 2):
            rest = [s for s in S8 if s not in first]
            for more in combinations(rest, 2):
                cnt[(frozenset(first + more), frozenset(lead))] += 1
    mult = set(cnt.values())
    ok = (md == [P // A if P % A == 0 else None, Q // A, R // A] and P == A * md[0] and Q == A * md[1] and R == A * md[2]
          and mult == {C(3, 1)})
    report(4, ok, f"P/A={sp.Rational(P, A)}, Q/A={sp.Rational(Q, A)}, R/A={sp.Rational(R, A)}; 축소판 중복배수={mult}(=₃C₁), md={md}")


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
