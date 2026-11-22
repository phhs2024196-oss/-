#!/usr/bin/env python3
"""CAU-G20A (2020학년도 중앙대 수시 논술 자연계열Ⅰ 수학 문제 1~3) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
PREFIX = "CAU-G20A"
import re
import sys
import itertools
from fractions import Fraction as Fr

import sympy as sp

WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")



def _rotations():
    """정육면체 회전군(24개)을 3x3 부호 순열 행렬(행렬식 +1)로 생성."""
    import numpy as np
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for i in range(3):
                M[i, perm[i]] = sg[i]
            if round(np.linalg.det(M)) == 1:
                out.append(M)
    return out


def p01():
    """주사위 두 개를 6끼리 붙이는 모든 배치(16가지=종류 4×회전 4) 전수, 면적 비례 확률 → P(새 ≤ 일반) (답 17/60)."""
    import numpy as np
    # 문제 그림의 주사위: 위 2, 앞 3, 오른쪽 1 (마주보는 면 합 7)
    base = {(0, 0, 1): 2, (0, -1, 0): 3, (1, 0, 0): 1}
    for v, k in list(base.items()):
        base[tuple(-c for c in v)] = 7 - k
    def placed(M):
        return {tuple(int(c) for c in M @ np.array(v)): k for v, k in base.items()}
    rots = _rotations()
    d1 = [placed(M) for M in rots if placed(M)[(1, 0, 0)] == 6]   # 6이 +x(접합면)
    d2 = [placed(M) for M in rots if placed(M)[(-1, 0, 0)] == 6]  # 6이 -x
    assert len(d1) == 4 and len(d2) == 4
    long_dirs = [(0, 1, 0), (0, 0, 1), (0, -1, 0), (0, 0, -1)]   # 순환 순서
    kinds = set()
    total = Fr(0)
    for A in d1:
        for B in d2:
            sums = [A[v] + B[v] for v in long_dirs]
            # 종류 = 긴 면 합의 순환열(회전·뒤집기 동일시)
            cyc = [tuple(sums[i:] + sums[:i]) for i in range(4)]
            cyc += [tuple(reversed(c)) for c in cyc]
            kinds.add(min(cyc))
            faces = [(s, Fr(2, 10)) for s in sums] + [(A[(-1, 0, 0)], Fr(1, 10)), (B[(1, 0, 0)], Fr(1, 10))]
            p = sum(w * Fr(sum(1 for m in range(1, 7) if s <= m), 6) for s, w in faces)
            total += p
    P = total / 16
    report(1, P == Fr(17, 60) and len(kinds) == 4, f"종류 {len(kinds)}가지 {sorted(kinds)}, P={P}≈{float(P):.4f}")


def p02():
    """(2-1) f=g(1)/g(x), h=F(f)-F(1), 최댓값 (170^{3/2}-2^{3/2})/3 / (2-2) b=1/(a-1)·(2,5), a+b^2 최솟값·최댓값."""
    g = sp.exp(t**2) * (t**2 + 3 * t + sp.Rational(5, 2))
    assert all(sp.simplify(g.subs(t, tv)) > 0 for tv in (-3, -1, 0, 2))  # g>0 (판별식 9-10<0)
    X = sp.symbols("X", real=True)
    f = sp.exp(-sp.integrate(sp.diff(g, t) / g, (t, 1, X)))
    f = sp.simplify(f)
    ok_f = sp.simplify(f - g.subs(t, 1) / g.subs(t, X)) == 0
    # g 최솟값: g'=0 실근
    crit = [r for r in sp.solve(sp.diff(g, t), t) if r.is_real]
    gmin_t = min(crit, key=lambda r: g.subs(t, r))
    fmax = sp.simplify(g.subs(t, 1) / g.subs(t, gmin_t))
    F = lambda u: (u**2 + 1) ** sp.Rational(3, 2) / 3
    hmax = sp.simplify(F(fmax) - F(1))
    # 수치: h(x)=int_1^x f f' sqrt(f^2+1) 직접 적분과 비교(x=-1), 격자 최대
    fx = sp.lambdify(X, f, "mpmath")
    import mpmath as mp
    fp = sp.lambdify(X, sp.diff(f, X), "mpmath")
    hnum = lambda xv: mp.quad(lambda s: fx(s) * fp(s) * mp.sqrt(fx(s) ** 2 + 1), [1, xv])
    grid = max(hnum(-4 + 0.05 * i) for i in range(0, 161))
    ok1 = ok_f and gmin_t == -1 and fmax == 13 and abs(float(hmax) - float(hnum(-1))) < 1e-8 and grid <= float(hmax) + 1e-9
    ok1 = ok1 and sp.simplify(hmax - (sp.Integer(170) ** sp.Rational(3, 2) - 2 ** sp.Rational(3, 2)) / 3) == 0
    # (2-2) 극한이 존재하는 (a,b) 전수: a in [3/2,3]
    a, b = sp.symbols("a b", real=True)
    expr = (x**3 + x**2 - a) / ((a - x) * (x + 1 - a))
    # 분모 x=1 에서 0 ⇔ a=1 또는 a=2; 구간 안은 a=2
    den1 = sp.solve(((a - 1) * (1 + 1 - a)), a)
    special = [r for r in den1 if sp.Rational(3, 2) <= r <= 3]
    b_special = [sp.limit(expr.subs(a, r), x, 1) for r in special]
    b_gen = sp.simplify(expr.subs(x, 1))
    vals = {}
    k = a + b_gen**2
    cand = [sp.Rational(3, 2), 3] + [r for r in sp.solve(sp.diff(k, a), a) if r.is_real and sp.Rational(3, 2) <= r <= 3 and r != 2]
    kvals = [sp.nsimplify(sp.simplify(k.subs(a, c))) for c in cand] + [r + bs**2 for r, bs in zip(special, b_special)]
    kmin = min(kvals, key=lambda v: float(v)); kmax = max(kvals, key=lambda v: float(v))
    grid2 = [float(k.subs(a, 1.5 + 0.0015 * i)) for i in range(0, 1001) if abs(1.5 + 0.0015 * i - 2) > 1e-9]
    ok2 = special == [2] and b_special == [5] and sp.simplify(b_gen - 1 / (a - 1)) == 0
    ok2 = ok2 and sp.simplify(kmin - (1 + 2 ** sp.Rational(1, 3) + 2 ** sp.Rational(-2, 3))) == 0 and kmax == 27
    ok2 = ok2 and min(grid2) >= float(kmin) - 1e-9 and max(grid2) < 27
    report(2, ok1 and ok2, f"(2-1) f=g(1)/g(x), f_max={fmax} at x={gmin_t}, h_max={hmax}≈{float(hmax):.4f}  (2-2) b={b_gen}, (a,b)=(2,{b_special[0]}), min={kmin}≈{float(kmin):.4f}, max={kmax}")


def p03():
    """(3-1) e^x 와 a sin x 접함 → a=√2e^{π/4}, 넓이 (2-√2)e^{π/4}-1 / (3-2) 평면-구 접함, 부피 V(a) 최소 a=(17+√145)/8."""
    al = sp.symbols("alpha", real=True)
    A = sp.symbols("A", positive=True)
    # 접점: e^α = A sin α, e^α = A cos α → tan α = 1, α ∈ (0, π/2]
    sol = [s for s in sp.solve(sp.tan(al) - 1, al) if 0 < s <= sp.pi / 2]
    alpha = sol[0]
    Aval = sp.simplify(sp.exp(alpha) / sp.sin(alpha))
    # 교점이 오직 하나인지 수치 확인: e^x - A sin x >= 0, 등호는 α에서만 (x∈[0,10])
    fn = sp.lambdify(x, sp.exp(x) - Aval * sp.sin(x))
    vals = [fn(0.001 * i) for i in range(0, 10001)]
    zeros = sum(1 for v in vals if v < 1e-6)
    area = sp.simplify(sp.integrate(sp.exp(x) - Aval * sp.sin(x), (x, 0, alpha)))
    ok1 = alpha == sp.pi / 4 and sp.simplify(Aval - sp.sqrt(2) * sp.exp(sp.pi / 4)) == 0 and min(vals) > -1e-9 and zeros <= 3
    ok1 = ok1 and sp.simplify(area - ((2 - sp.sqrt(2)) * sp.exp(sp.pi / 4) - 1)) == 0
    # (3-2)
    a, b = sp.symbols("a b", positive=True)
    # 평면 x/a + y/(2a) + z/b = 1 과 중심 (1,1,1) 거리 1
    nrm = sp.Matrix([1 / a, 1 / (2 * a), 1 / b])
    dist2 = (nrm.dot(sp.Matrix([1, 1, 1])) - 1) ** 2 - nrm.dot(nrm)
    bs = [s for s in sp.solve(sp.numer(sp.together(dist2)), b)]
    bs = [s for s in bs if s.subs(a, 4) > 0]
    V = sp.simplify(a * 2 * a * bs[0] / 6)
    crit = [r for r in sp.solve(sp.numer(sp.together(sp.diff(V, a))), a) if r.is_real and r > 3]
    Vf = sp.lambdify(a, V)
    grid = min((Vf(3.001 + 0.001 * i), 3.001 + 0.001 * i) for i in range(0, 20000))
    ok2 = len(crit) == 1 and sp.simplify(crit[0] - (17 + sp.sqrt(145)) / 8) == 0 and abs(grid[1] - float(crit[0])) < 2e-3
    ok2 = ok2 and sp.simplify(V - a**3 * (2 * a - 3) / (3 * (a**2 - 3 * a + 1))) == 0
    report(3, ok1 and ok2, f"(3-1) α={alpha}, a={Aval}, 넓이={area}≈{float(area):.4f}  (3-2) b={bs[0]}, V={V}, a*={crit[0]}≈{float(crit[0]):.4f} (격자 {grid[1]:.3f})")

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

