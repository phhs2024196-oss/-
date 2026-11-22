#!/usr/bin/env python3
"""AJOU-G26B (아주대 2026학년도 논술고사 자연계열(오후)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "AJOU-G26B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def md_pairs(no: int, sub: str) -> dict:
    """'k=3; limit=1/2' 또는 'radius: ..; circle: ..' 꼴을 사전으로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = re.split(r"[:=]", seg, maxsplit=1)
        out[k.strip()] = v.strip()
    return out


def md_set(no: int, sub: str) -> list:
    return [sp.sympify(s) for s in md_answer(no, sub).strip().strip("{}").split(",")]


def same(a, b, tol=1e-10) -> bool:
    return abs(complex(sp.N(a, 30)) - complex(sp.N(b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


X, Y = sp.symbols("X Y", real=True)


def _P(nv, i):
    th = sp.pi / 2 - 2 * sp.pi * (i - 1) / nv
    return sp.Matrix([sp.cos(th), sp.sin(th)])


def _foot(Q, A, B):
    d = B - A
    return A + ((Q - A).dot(d) / d.dot(d)) * d


def _circle_through(pts):
    a, b, r2 = sp.symbols("a b r2")
    s = sp.solve([(p_[0] - a)**2 + (p_[1] - b)**2 - r2 for p_ in pts], [a, b, r2], dict=True)[0]
    return s[a], s[b], s[r2]


def _meet(A1, A2, B1, B2):
    u, v = sp.symbols("u v")
    s = sp.solve(list(A1 + u * (A2 - A1) - B1 - v * (B2 - B1)), [u, v])
    return A1 + s[u] * (A2 - A1)


def p01():
    # (1) n=8
    O = sp.Matrix([0, 0])
    Hs = [_foot(O, _P(8, 1), _P(8, j)) for j in (2, 3, 4)]
    a, b, r2 = _circle_through(Hs)
    mine = (X - a)**2 + (Y - b)**2 - r2
    md = sp.sympify(md_answer(1, "1-1"), locals={"x": X, "y": Y})
    ok1 = sp.simplify(sp.expand(mine - md)) == 0
    # (2) n=10, k=1..4 의 원 넓이 → 공비 확인 후 급수
    def Sk(k):
        Q = sp.Matrix([0, 1 - 3 * sp.Rational(2, 9)**k])
        return sp.pi * sp.nsimplify(sp.N(_circle_through([_foot(Q, _P(10, 1), _P(10, j)) for j in (2, 3, 4)])[2], 40))
    S = [Sk(k) for k in (1, 2, 3, 4)]
    ratio = sp.simplify(S[1] / S[0])
    geo = all(sp.simplify(S[i + 1] / S[i] - ratio) == 0 for i in range(3))
    total = S[0] / (1 - ratio)
    ok2 = geo and same(total, sp.sympify(md_answer(1, "1-2")))
    # (3) n=12, 사각형 P2 A B P3 (신발끈)
    A = _meet(_P(12, 1), _P(12, 4), _P(12, 2), _P(12, 5))
    B = _meet(_P(12, 1), _P(12, 4), _P(12, 3), _P(12, 6))
    pts = [_P(12, 2), A, B, _P(12, 3)]
    area = sp.Abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))) / 2
    ok3 = same(area, sp.sympify(md_answer(1, "1-3")))
    # (2-1) n=16
    nv = 16
    def m_(j):
        return X / sp.Rational(j, nv) + Y / sp.Rational(nv - j, nv) - 1
    ps = [sp.solve([m_(8), m_(j)], [X, Y])[X] for j in list(range(1, 8)) + list(range(9, 16))]
    ok4 = sum(ps) == sp.sympify(md_answer(1, "2-1"))
    # (2-2) 교점 y좌표를 일반 i,n 으로 구해 합, 극한. 작은 n 은 직접 구성과 대조
    i, N = sp.symbols("i N", positive=True, integer=True)
    sol = sp.solve([X * N / i + Y * N / (N - i) - 1, X * N / (i - 1) + Y * N / (N - i + 1) - 1], [X, Y], dict=True)[0]
    yi = sp.factor(sol[Y])
    Tn = sp.Rational(1, 2) / N * (N - 1) / N + sp.summation(yi / (2 * N), (i, 2, N - 1))
    def Tdirect(nv):
        tot = sp.Rational(1, 2) * sp.Rational(1, nv) * sp.Rational(nv - 1, nv)
        for iv in range(2, nv):
            s_ = sp.solve([X * nv / iv + Y * nv / (nv - iv) - 1, X * nv / (iv - 1) + Y * nv / (nv - iv + 1) - 1], [X, Y])
            tot += sp.Rational(1, 2) * sp.Rational(1, nv) * s_[Y]
        return tot
    cross = all(sp.simplify(Tn.subs(N, nv) - Tdirect(nv)) == 0 for nv in (3, 5, 8))
    ok5 = cross and same(sp.limit(Tn, N, sp.oo), sp.sympify(md_answer(1, "2-2")))
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"circle={sp.expand(mine)} ΣS={sp.nsimplify(total)} area={sp.nsimplify(sp.N(area, 40), [sp.sqrt(3)])} "
           f"Σp={sum(ps)} limT={sp.limit(Tn, N, sp.oo)} [{ok1},{ok2},{ok3},{ok4},{ok5}]")


def p02():
    import numpy as np
    # (1) k=7, 0≤t≤2π
    k = 7
    ts = np.linspace(0, 2 * np.pi, 2_000_001)
    OA = np.hypot(k / 2 * np.cos(ts) + np.sin(5 * ts), k / 2 * np.sin(ts) + np.cos(5 * ts))
    OB = np.hypot(k / 2 * np.cos(ts) + np.cos(5 * ts), k / 2 * np.sin(ts) + np.sin(5 * ts))
    r = OA.min()
    def nmin(arr):
        idx = np.where((arr[1:-1] <= arr[:-2]) & (arr[1:-1] <= arr[2:]) & (np.abs(arr[1:-1] - r) < 1e-6))[0]
        return len(idx) + int(abs(arr[0] - r) < 1e-9) + int(abs(arr[-1] - r) < 1e-9 and abs(arr[0] - r) >= 1e-9)
    mn = nmin(OA) * nmin(OB)
    ok1 = mn == int(sp.sympify(md_answer(2, "1")))
    # (2-1) k=10, t1 = AB 최대인 가장 작은 양수
    tt = sp.symbols("tt", real=True)
    AB2 = sp.expand_trig((sp.sin(5 * tt) - sp.cos(5 * tt))**2 * 2)
    grid = np.linspace(1e-6, np.pi, 300001)
    AB2f = sp.lambdify(tt, AB2, "numpy")(grid)
    t1 = grid[int(np.argmax(AB2f >= AB2f.max() - 1e-9))]          # 최대가 되는 가장 작은 t
    t1 = float(sp.nsolve(sp.diff(AB2, tt), tt, t1, prec=30))     # 격자 근사를 뉴턴법으로 정밀화
    Pp = np.array([5 * np.cos(t1), 5 * np.sin(t1)])
    A = Pp + np.array([np.sin(5 * t1), np.cos(5 * t1)])
    B = Pp + np.array([np.cos(5 * t1), np.sin(5 * t1)])
    d = B - A
    us = np.roots([d @ d, 2 * A @ d, A @ A - 25])
    Q, T = [A + u * d for u in us]
    QT2 = float(np.sum((Q - T)**2))
    sols = [(a, b, c) for a in range(1, 400) for b in range(1, 6) for c in range(1, 400)
            if abs(a * (b - math.sin(3 * math.pi / c)) - QT2) < 1e-6]
    ok2 = len({a + b + c for a, b, c in sols}) == 1 and sum(sols[0]) == int(sp.sympify(md_answer(2, "2-1")))
    # (2-2)
    ang = math.acos(float(Q @ T) / 25)
    val = 12.5 * ang - 12.5 * math.sin(ang)
    ok3 = abs(val - float(sp.N(sp.sympify(md_answer(2, "2-2"))))) < 1e-6
    # (3-1) h(t)=y/x 가 (0,π/2) 에서 증가하는 3보다 큰 최소 k
    kk = sp.symbols("k", positive=True)
    xb = kk / 2 * sp.cos(tt) + sp.cos(5 * tt)
    yb = kk / 2 * sp.sin(tt) + sp.sin(5 * tt)
    num = sp.lambdify((kk, tt), sp.diff(yb, tt) * xb - yb * sp.diff(xb, tt), "numpy")
    den = sp.lambdify((kk, tt), xb, "numpy")
    g2 = np.linspace(1e-6, np.pi / 2 - 1e-6, 200001)
    def inc(kv):
        return np.all(num(kv, g2) >= -1e-9) and np.all(np.abs(den(kv, g2)) > 1e-9)
    kmin1 = next(kv for kv in range(4, 50) if inc(kv))
    ok4 = kmin1 == int(sp.sympify(md_answer(2, "3-1")))
    # (3-2) dx/dt ≠ 0 on (0,π/2) 인 10보다 큰 최소 k
    dxdt = sp.lambdify((kk, tt), sp.diff(xb, tt), "numpy")
    g3 = np.linspace(1e-7, np.pi / 2 - 1e-7, 2_000_001)
    def defined(kv):
        v = dxdt(kv, g3)
        return np.all(v < 0)                      # 부호가 바뀌면(0을 지나면) 정의되지 않음
    kmin2 = next(kv for kv in range(11, 60) if defined(kv))
    ok5 = kmin2 == int(sp.sympify(md_answer(2, "3-2")))
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"mn={mn} t1≈{t1:.6f} QT²={QT2:.9f} (a,b,c)={sols} S2-S1={val:.9f} k1={kmin1} k2={kmin2} "
           f"[{ok1},{ok2},{ok3},{ok4},{ok5}]")


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
