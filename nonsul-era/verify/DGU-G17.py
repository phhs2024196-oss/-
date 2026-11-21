#!/usr/bin/env python3
"""DGU-G17 (2017학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G17"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p03():
    """구 부피(적분) + 두 쌓기 방식의 밀도를 격자 좌표로 직접 계산해 비교."""
    r, X = sp.symbols("r X", positive=True)
    Vs = 2 * sp.integrate(sp.pi * (r**2 - X**2), (X, 0, r))
    ok_v = sp.simplify(Vs - sp.Rational(4, 3) * sp.pi * r**3) == 0
    # 철수: 1층 (2ri,2rj,0), 2층 (2ri+r,2rj+r,h). 2층 구가 아래 4개와 접함 → h
    h = sp.symbols("h", positive=True)
    hc = sp.solve(sp.Eq(r**2 + r**2 + h**2, (2 * r)**2), h)[0]
    dens_c = sp.simplify(2 * Vs / ((2 * r) * (2 * r) * (2 * hc)))   # 단위 기둥 2r×2r×2h 에 구 2개
    # 영희: 정육각형(삼각) 배열, 구 1개당 넓이 2√3 r², 층 간격 = 정사면체 높이
    hy = sp.solve(sp.Eq((2 * r / sp.sqrt(3))**2 + h**2, (2 * r)**2), h)[0]
    dens_y = sp.simplify(Vs / (2 * sp.sqrt(3) * r**2 * hy))
    # 겹침 없음 확인(수치, r=1): 철수 3층 격자 최소 거리 ≥ 2
    pts = []
    for k in range(3):
        off = 1.0 if k % 2 else 0.0
        for i in range(-3, 4):
            for j in range(-3, 4):
                pts.append((2 * i + off, 2 * j + off, k * float(hc.subs(r, 1))))
    P = np.array(pts)
    d = np.sqrt(((P[:, None, :] - P[None, :, :])**2).sum(-1))
    dmin = d[d > 1e-9].min()
    # 재수록본 풀이1(체심입방 가정) 값 √3π/8 은 실제 배열과 다름
    bcc = sp.sqrt(3) * sp.pi / 8
    ok = ok_v and sp.simplify(dens_c - dens_y) == 0 and sp.simplify(dens_c - sp.sqrt(2) * sp.pi / 6) == 0 \
        and abs(dmin - 2) < 1e-9 and sp.simplify(bcc - dens_c) != 0
    report(3, ok, f"V=4πr³/3, 철수 h=√2r 밀도={dens_c}, 영희 밀도={dens_y} → 같음(최소 중심거리 {dmin:.6f}); 풀이1의 √3π/8 ≠")


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
