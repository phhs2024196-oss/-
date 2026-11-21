#!/usr/bin/env python3
"""KHU-M23D (경희대 2023학년도 모의논술 의·약학계 수학 논제 I-1, I-2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import math
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "KHU-M23D"
WIDTH = 2
RESULTS = []
th = sp.symbols("theta", positive=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) 가로 3 방향 단면: 아래·위 공 중심 x=1, 가운데 공 x=3-1=2, 반지름 1, 외접 ⇒ 중심 거리 2
    dx = 2 - 1
    dz = sp.sqrt(2**2 - dx**2)
    H = 1 + 2 * dz + 1
    ok1 = sp.simplify(H - (2 + 2 * sp.sqrt(3))) == 0
    # (2) 중심(단면 좌표 x,z), 광선 방향 (-cos30, -sin30)
    cs = [(1.0, 1.0), (2.0, 1.0 + math.sqrt(3)), (1.0, 1.0 + 2 * math.sqrt(3))]
    nrm = np.array([-math.sin(math.pi / 6), math.cos(math.pi / 6)])  # 광선에 수직
    proj = sorted(float(np.dot(nrm, c)) for c in cs)
    gaps = [proj[1] - proj[0], proj[2] - proj[1]]

    def lens(dist):
        return 2 * math.acos(dist / 2) - dist / 2 * math.sqrt(4 - dist**2) if dist < 2 else 0.0
    union = 3 * math.pi - sum(lens(g) for g in gaps)
    # 격자로 세 원(단위원, 중심 proj, 같은 직선 위) 합집합 넓이 확인
    xs = np.linspace(proj[0] - 1.01, proj[2] + 1.01, 3001)
    ys = np.linspace(-1.01, 1.01, 1501)
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros_like(X, dtype=bool)
    for c in proj:
        inside |= (X - c)**2 + Y**2 <= 1
    grid = inside.mean() * (xs[-1] - xs[0]) * (ys[-1] - ys[0])
    shadow = union / math.cos(math.pi / 3)   # 광선에 수직인 면과 지면이 이루는 각 60°
    target = float(sp.Rational(14, 3) * sp.pi + sp.sqrt(3))
    ok2 = abs(shadow - target) < 1e-12 and abs(grid - union) < 2e-2
    report(1, ok1 and ok2, f"(1) 높이={H}; (2) 투영 중심 간격={[round(g, 6) for g in gaps]}, 단면={union:.6f}(격자 {grid:.4f}), 그림자={shadow:.6f}")


def p02():
    s = sp.sin(th)
    # (1) 부채꼴 OA=1, 반각 θ: 첫 원 r1 = sinθ(1-r1)
    r = sp.symbols("r", positive=True)
    r1 = sp.solve(sp.Eq(r, s * (1 - r)), r)[0]
    # 이웃 원: 중심이 이등분선 위, OC = r/sinθ, 접하면 OC_n - OC_{n+1} = r_n + r_{n+1}
    rn, rn1 = sp.symbols("rn rn1", positive=True)
    ratio = sp.simplify(sp.solve(sp.Eq(rn / s - rn1 / s, rn + rn1), rn1)[0] / rn)
    total = sp.simplify(sp.pi * r1**2 / (1 - ratio**2))
    ok1 = sp.simplify(total - sp.pi * s / 4) == 0
    # 수치: θ=0.4 에서 원을 실제로 200개 쌓아 넓이 합
    tv = 0.4
    rr = math.sin(tv) / (1 + math.sin(tv)); acc = 0.0
    oc = 1 - rr
    for _ in range(400):
        acc += math.pi * rr**2
        # 다음 원: oc' = rr'/sin, oc - oc' = rr + rr'
        rr2 = (oc - rr) / (1 / math.sin(tv) + 1); oc = rr2 / math.sin(tv); rr = rr2
    ok1 &= abs(acc - math.pi * math.sin(tv) / 4) < 1e-12
    # (2) r100 = r1 * ratio^99, x=sinθ 치환
    xv = sp.symbols("x", positive=True)
    g = (r1 * ratio**99).subs(s, xv)
    g = (r1 * ratio**99).subs(sp.sin(th), xv)
    crit = [c for c in sp.solve(sp.diff(g, xv), xv) if c.is_real and 0 < c < 1]
    ok2 = len(crit) == 1
    x0 = crit[0]
    ok2 &= sp.diff(g, xv).subs(xv, x0 / 2) > 0 and sp.diff(g, xv).subs(xv, (1 + x0) / 2) < 0
    r1v = sp.simplify(r1.subs(sp.sin(th), x0))
    ok2 &= r1v == sp.Rational(1, 200)
    report(2, ok1 and ok2, f"(1) 합={total}; (2) sinθ={x0}, r1={r1v}")


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
