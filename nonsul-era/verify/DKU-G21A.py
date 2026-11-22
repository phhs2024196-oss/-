"""DKU-G21A 검산: 문제 조건에서 독립 재계산한 값을 items/DKU/PAST 의 answer.value 와 비교."""
PREFIX = "DKU-G21A"
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


def f21(xv):
    xv = mp.mpf(xv)
    if xv < -1:
        return (xv + 3) / (2 * mp.e)
    if xv < 1:
        return (2 * xv**2 - 1) * mp.e ** (-xv**2)
    return -(xv - 3) / (2 * mp.e)


_F0 = {}


def F0(xv):
    k = round(float(xv), 12)
    if k not in _F0:
        pts = [-3, -1, 0, 1, 3]
        lo = [p for p in pts if p < xv]
        _F0[k] = mp.quad(f21, sorted(set([0] + [p for p in pts if 0 < p < xv] + [xv]))) if xv >= 0 else \
            -mp.quad(f21, sorted(set([xv] + [p for p in pts if xv < p < 0] + [0])))
    return _F0[k]


def p01():
    # 논제1
    v1 = mp.quad(f21, [0, 1])
    ok1 = abs(v1 - sym(md_answer(1, "1")).evalf(30)) < 1e-12
    sub(ok1, f"논제1 numeric={mp.nstr(v1, 12)}")
    # 논제2: F(x)=F0(x)-F0(a). F0 의 단조 구간에서 해의 개수를 센다
    r = mp.findroot(lambda z: 2 * z**2 - 1, 0.7)  # f=0 인 점(양수)
    br = [mp.mpf(-3), -r, r, mp.mpf(3)]
    vals = [F0(b) for b in br]

    def count(a):
        v = F0(a)
        roots = []
        for i in range(3):
            lo, hi = br[i], br[i + 1]
            flo, fhi = vals[i], vals[i + 1]
            if min(flo, fhi) - 1e-12 <= v <= max(flo, fhi) + 1e-12:
                if abs(v - flo) < 1e-12:
                    z = lo
                elif abs(v - fhi) < 1e-12:
                    z = hi
                else:
                    z = mp.findroot(lambda s: F0(s) - v, (lo, hi), solver="anderson")
                if all(abs(z - w) > 1e-7 for w in roots):
                    roots.append(z)
        return len(roots)

    gen = [mp.mpf(-3) + mp.mpf(6) * i / 120 + mp.mpf(1) / 997 for i in range(120)]
    generic_one = [a for a in gen if count(a) == 1]
    special = []
    for v in vals:
        for i in range(3):
            lo, hi = br[i], br[i + 1]
            flo, fhi = vals[i], vals[i + 1]
            if min(flo, fhi) - 1e-12 <= v <= max(flo, fhi) + 1e-12:
                if abs(v - flo) < 1e-12:
                    special.append(lo)
                elif abs(v - fhi) < 1e-12:
                    special.append(hi)
                else:
                    special.append(mp.findroot(lambda s: F0(s) - v, (lo, hi), solver="anderson"))
    cand = []
    for a in special:
        if count(a) == 1 and all(abs(a - c) > 1e-7 for c in cand):
            cand.append(a)
    cand = sorted(float(c) for c in cand)
    exp = sym_set(md_answer(1, "2"))
    ok2 = not generic_one and len(cand) == len(exp) and all(abs(a - b) < 1e-6 for a, b in zip(cand, exp))
    sub(ok2, f"논제2 a={[round(c, 6) for c in cand]} generic_one={len(generic_one)}")

    # 논제3
    def g(z):
        return z - z * z + abs(z - z * z)

    def h(z):
        return 16 * g(z / 4) + 8 * g(z / 2 - 2)

    def phi(z):
        return h(z) / z

    def nroots(tv):
        c = phi(tv)
        if abs(c) < 1e-12:
            return 10**9
        rs = [0.0]
        z = (8 - c) / 2  # 구간 (0,4] 에서 8-2x=c ... h/x 를 수치 탐색으로 확인
        xs = np.linspace(-1.000031, 7.000077, 80001)
        xs = xs[np.abs(xs) > 1e-9]
        vv = np.array([phi(float(s)) - c for s in xs])
        cnt = 0
        for i in range(len(xs) - 1):
            if xs[i] < 0 < xs[i + 1]:
                continue
            if vv[i] * vv[i + 1] < 0:
                cnt += 1
        # 접하는 근(부호가 안 바뀜): |vv| 의 국소 최소이면서 거의 0
        av = np.abs(vv)
        for i in range(1, len(xs) - 1):
            if av[i] < 1e-6 and av[i] <= av[i - 1] and av[i] <= av[i + 1] and vv[i - 1] * vv[i + 1] >= 0:
                cnt += 1
        return 1 + cnt

    # 격자 + 접점 후보. 접점은 h(x)/x 의 두 번째 봉우리 최댓값
    xm = mp.findroot(lambda z: mp.diff(phi, z), 5.0)
    M = phi(xm)
    tv = [mp.findroot(lambda z: phi(z) - M, 3.5), xm]
    tv.sort()
    got = sorted(float(v) for v in tv)
    exp3 = sym_set(md_answer(1, "3"))
    ok3 = len(got) == len(exp3) and all(abs(a - b) < 1e-7 for a, b in zip(got, exp3))
    # 격자(접점 제외)에서는 근의 개수가 3 이 되는 t 가 없고, 후보에서는 3
    grid_ts = [0.05 + 5.9 * i / 59 for i in range(60)]
    grid3 = [s for s in grid_ts if all(abs(s - c) > 0.03 for c in got) and nroots(s) == 3]
    cand_cnt = [nroots(c) for c in got]
    # 접점 근방은 수치 판별이 민감하므로 접점이 아닌 후보(첫째)만 근 개수로 직접 확인
    ok3 = ok3 and not grid3 and cand_cnt == [3, 3]
    sub(ok3, f"논제3 t={[round(v, 6) for v in got]} grid3={grid3} cnt={cand_cnt}")
    flush(1)


def p02():
    p, q = Fraction(1, 5), Fraction(4, 5)
    row = [Fraction(1)]
    for _ in range(100):
        new = [Fraction(0)] * (len(row) + 1)
        for i, pr in enumerate(row):
            new[i] += pr * p
            new[i + 1] += pr * q
        row = new
    # 번호 x 는 왼쪽부터이므로 row[x] 가 P(X=x)
    ratio = row[87] / row[88]
    ok1 = Fraction(md_answer(2, "1")) == ratio
    sub(ok1, f"논제1 P87/P88={ratio}")
    am = max(range(len(row)), key=lambda i: row[i])
    ok2 = int(md_answer(2, "2")) == am and sum(row) == 1
    sub(ok2, f"논제2 argmax={am}")
    flush(2)


if __name__ == "__main__":
    main(globals())
