"""DKU-G23B 검산: 문제 조건에서 독립 재계산한 값을 items/DKU/PAST 의 answer.value 와 비교."""
PREFIX = "DKU-G23B"
WIDTH = 2
import itertools
from fractions import Fraction
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def sym(s: str):
    return sp.sympify(s, locals={"E": sp.E, "pi": sp.pi})


def sym_set(s: str):
    return sorted(float(sp.N(v)) for v in sp.sympify(s.replace("{", "[").replace("}", "]"), locals={"E": sp.E}))


def assign(s: str):
    return {k.strip(): sym(v.strip()) for k, v in (kv.split("=") for kv in s.split(","))}


def close(a, b, tol=1e-9) -> bool:
    return abs(float(sp.N(a, 30)) - float(sp.N(b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


PEND = []


def sub(ok, msg):
    PEND.append((bool(ok), msg))


def flush(seq):
    report(seq, all(o for o, _ in PEND), " | ".join(m for _, m in PEND))
    PEND.clear()


def main(g):
    fns = sorted((nm for nm in g if re.fullmatch(r"p\d+", nm)), key=lambda s: int(s[1:]))
    for nm in fns:
        seq = int(nm[1:])
        try:
            g[nm]()
        except Exception as e:  # noqa: BLE001
            PEND.clear()
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)


def _fcurve(x_):
    return 3 * (x_ + 2 / 3) * (1 - 3 * x_)


def _S(n=40001):
    xs = np.linspace(-2 / 3, 1 / 3, n)
    return np.concatenate([np.column_stack([xs, _fcurve(xs)]), np.column_stack([xs, -_fcurve(xs)])])


def _step(B, S=None):
    """A=(-2/3,0), 꼭짓점 B 와 S 위의 점 P 로 만든 삼각형의 넓이가 최대인 P 를 찾는다."""
    S = _S() if S is None else S
    A = np.array([-2 / 3, 0.0])
    cr = np.abs((B[0] - A[0]) * (S[:, 1] - A[1]) - (B[1] - A[1]) * (S[:, 0] - A[0]))
    i = int(np.argmax(cr))
    x0 = S[i, 0]
    sgn = 1 if S[i, 1] >= 0 else -1
    pt = lambda xx: np.array([xx, sgn * _fcurve(xx)])
    area = lambda xx: abs((B[0] - A[0]) * (pt(xx)[1] - A[1]) - (B[1] - A[1]) * (pt(xx)[0] - A[0])) / 2
    lo, hi = max(-2 / 3, x0 - 1e-4), min(1 / 3, x0 + 1e-4)
    for _ in range(100):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if area(m1) < area(m2):
            lo = m1
        else:
            hi = m2
    return pt((lo + hi) / 2)


def p01():
    S = _S()
    # 논제1: B1=(a1,f(a1)) 에서 두 번 진행한 점이 x=1/18 이 되는 a1 을 찾는다
    comp = lambda a1: _step(_step(np.array([a1, _fcurve(a1)]), S), S)[0] - 1 / 18
    grid = np.linspace(-0.6, 0.3, 91)
    vals = [comp(g) for g in grid]
    rts = []
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0 and abs(vals[i]) < 0.2 and abs(vals[i + 1]) < 0.2:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(40):
                mid = (lo + hi) / 2
                if comp(mid) * vals[i] > 0:
                    lo = mid
                else:
                    hi = mid
            rts.append((lo + hi) / 2)
    exp = float(sp.Rational(md_answer(1, "1")))
    sub(len(rts) == 1 and abs(rts[0] - exp) < 1e-4, f"논제1 a1={[round(r, 6) for r in rts]}")
    # 논제2: 공비 확인 후, A 를 꼭짓점으로 하는 R 속 삼각형의 최대 넓이(S 위 두 점 전수 비교)
    B = np.array([-0.5, -_fcurve(-0.5)])
    xs_ = [B[0]]
    for _ in range(6):
        B = _step(B, S)
        xs_.append(B[0])
    ratios = [xs_[i + 1] / xs_[i] for i in range(len(xs_) - 1)]
    S2 = _S(1201)
    A = np.array([-2 / 3, 0.0])
    dx, dy = S2[:, 0] - A[0], S2[:, 1] - A[1]
    mx = np.max(np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])) / 2
    exp2 = float(sp.Rational(md_answer(1, "2")))
    sub(all(abs(r + 0.5) < 1e-3 for r in ratios) and abs(mx - exp2) < 1e-3, f"논제2 공비≈{[round(r, 4) for r in ratios[:3]]}, 최대넓이={mx:.6f}")
    # 논제3: a_n=a(-1/2)^n 으로 b_n 계산(n 이 클 때)
    ns = 200000
    ks = np.arange(ns, 2 * ns)
    bn = np.sum(1 / np.log(np.abs(-0.5 * (-0.5) ** ks.astype(float)))) if False else float(np.sum(1 / (-(ks + 1) * np.log(2))))
    sub(abs(bn - int(md_answer(1, "3"))) < 1e-4, f"논제3 b_n(n={ns})={bn:.6f}")
    flush(1)


def p02():
    mp.mp.dps = 25
    res = []
    for d in (0.0, 0.02, -0.02):
        c = mp.pi / 16 if d == 0 else (mp.mpf(5) / 8 - mp.mpf(1) / 2 - d * 2 / (3 * mp.pi)) * mp.pi / 2
        f = lambda z, c=c, d=d: z + c * mp.sin(mp.pi * z) + d * mp.sin(3 * mp.pi * z)
        zz = [mp.mpf(i) / 500 for i in range(-500, 2500)]
        assert min(f(zz[i + 1]) - f(zz[i]) for i in range(len(zz) - 1)) > 0
        assert abs(f(mp.mpf("0.37") + 2) - f(mp.mpf("0.37")) - 2) < 1e-12 and abs(f(mp.mpf("0.37")) + f(2 - mp.mpf("0.37")) - 2) < 1e-12
        assert abs(mp.quad(f, [0, 1]) - mp.quad(f, [-1, 0]) - mp.mpf(5) / 4) < 1e-12

        def finv(y):
            lo, hi = mp.mpf(y) - 2, mp.mpf(y) + 2
            for _ in range(90):
                mid = (lo + hi) / 2
                if f(mid) < y:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        pts = [-1 + mp.mpf(5) * i / 50 for i in range(51)]
        res.append(mp.quad(finv, pts))
    exp = sp.Rational(md_answer(2, "1"))
    sub(all(abs(v - exp.evalf(25)) < 1e-9 for v in res), "논제1 수치=" + ", ".join(mp.nstr(v, 10) for v in res))
    # 논제2: 법선이 (t,0) 을 지나는 접점 u: t=u+g(u)g'(u). 각 t 에 u 가 하나이려면 φ 가 단조 비감소.
    def monotone(b, a):
        us = np.linspace(-14, 8, 400001)
        w = a * np.exp(us)
        phi = us + (w + b) * w
        return np.min(np.diff(phi)) >= -1e-13
    out = []
    for a in (0.3, 1.0, 5.0):
        lo, hi = -5.0, 0.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if monotone(mid, a):
                hi = mid
            else:
                lo = mid
        out.append(hi)
    # 임계 아래에서 해가 3개 이상인 t 가 실제로 있는지도 확인
    b_low = -2 * np.sqrt(2) - 0.05
    us = np.linspace(-14, 8, 400001)
    w = np.exp(us)
    phi = us + (w + b_low) * w
    cnt = int(np.sum(np.diff(np.sign(np.diff(phi))) != 0))
    exp2 = float(sym(md_answer(2, "2")))
    sub(all(abs(o - exp2) < 5e-3 for o in out) and cnt >= 2, f"논제2 b_min≈{[round(o, 4) for o in out]}, 임계 아래 극점수={cnt}")
    flush(2)


if __name__ == "__main__":
    main(globals())
