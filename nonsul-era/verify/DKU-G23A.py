"""DKU-G23A 검산: 문제 조건에서 독립 재계산한 값을 items/DKU/PAST 의 answer.value 와 비교."""
PREFIX = "DKU-G23A"
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


def _bps(a):
    """f0=(x^3-a^2 x)/2 가 ±3 이 되는 서로 다른 실수 x (정렬)."""
    rs = []
    for c in (6.0, -6.0):
        for r in np.roots([1, 0, -a * a, c]):
            if abs(r.imag) < 1e-9:
                rs.append(r.real)
    rs.sort()
    out = []
    for r in rs:
        if not out or abs(r - out[-1]) > 1e-9:
            out.append(r)
    return out


def _selections(a, tails_const=True):
    """연속인 선택 f: 각 구간에서 {3, -3, f0} 중 하나. 구간 경계는 f0=±3 인 점."""
    f0 = lambda z: (z**3 - a * a * z) / 2
    bp = _bps(a)
    m = len(bp)
    names = ("3", "m", "f")

    def val(br, z):
        return 3.0 if br == "3" else (-3.0 if br == "m" else f0(z))
    out = []
    for sig in itertools.product(names, repeat=m + 1):
        if tails_const and (sig[0] == "f" or sig[-1] == "f"):
            continue
        if any(abs(val(sig[i], bp[i]) - val(sig[i + 1], bp[i])) > 1e-9 for i in range(m)):
            continue
        out.append(sig)
    return bp, out, f0


def _intervals(bp):
    ed = [-np.inf] + list(bp) + [np.inf]
    return [(ed[i], ed[i + 1]) for i in range(len(ed) - 1)]


def _stats(a, bp, sig, f0):
    zeros = 0
    hi, lo = -1e18, 1e18
    for (l, r), br in zip(_intervals(bp), sig):
        if br == "3":
            hi, lo = max(hi, 3.0), min(lo, 3.0)
        elif br == "m":
            hi, lo = max(hi, -3.0), min(lo, -3.0)
        else:
            cands = [l, r] + [c for c in (-a / np.sqrt(3), a / np.sqrt(3)) if l < c < r]
            for c in cands:
                if np.isfinite(c):
                    hi, lo = max(hi, f0(c)), min(lo, f0(c))
            zeros += sum(1 for z in (-a, 0.0, a) if l < z < r)
    return zeros, hi, lo


def p01():
    # 논제1: a 를 격자로 훑어 '근 1개 & 최대+최소=24' 가 되는 선택이 있는 a 를 찾는다
    def best(a):
        bp, sels, f0 = _selections(a)
        res = []
        for sg in sels:
            z, hi, lo = _stats(a, bp, sg, f0)
            if z == 1:
                res.append((sg, hi + lo - 24))
        return res
    roots = []
    prev = None
    ga = [0.2 + i * 0.0053 for i in range(2200)]
    for a in ga:
        cur = {sg: d for sg, d in best(a)}
        if prev is not None:
            for sg, d in cur.items():
                if sg in prev and prev[sg] * d < 0:
                    lo, hi = a - 0.0053, a
                    d0 = prev[sg]
                    for _ in range(60):
                        mid = (lo + hi) / 2
                        dm = dict(best(mid)).get(sg)
                        if dm is None:
                            break
                        if dm * d0 > 0:
                            lo = mid
                        else:
                            hi = mid
                    roots.append((lo + hi) / 2)
        prev = cur
    roots = sorted(set(round(r, 6) for r in roots))
    exp = float(sp.N(sym(md_answer(1, "1"))))
    sub(len(roots) == 1 and abs(roots[0] - exp) < 1e-5, f"논제1 a={roots}")
    # 논제2: (4) 로 a 의 후보를 정하고, 조건 (1)~(3) 을 모두 만족하는 f 의 합을 열거
    aa = sp.symbols("aa", positive=True)
    cand_a = sorted(set(sp.solve(sp.Eq((-27 + 3 * aa**2) / 2, 3), aa) + sp.solve(sp.Eq((-27 + 3 * aa**2) / 2, -3), aa)))
    found = {}
    for av in cand_a:
        a = float(av)
        bp, sels, f0 = _selections(a, tails_const=False)

        def fv(sig, z):
            for (l, r), br in zip(_intervals(bp), sig):
                if l - 1e-12 <= z <= r + 1e-12:
                    return 3.0 if br == "3" else (-3.0 if br == "m" else f0(z))

        def gcount(sig, k):
            pts = []
            for (l, r), br in zip(_intervals(bp), sig):
                if br in ("3", "m"):
                    c = 3.0 if br == "3" else -3.0
                    z = k + c / a**2
                    if l - 1e-9 <= z <= r + 1e-9:
                        pts.append(z)
                else:
                    for rt in np.roots([1, 0, -3 * a * a, 2 * a * a * k]):
                        if abs(rt.imag) < 1e-7 and l - 1e-9 <= rt.real <= r + 1e-9:
                            pts.append(rt.real)
            pts.sort()
            u = []
            for z in pts:
                if not u or abs(z - u[-1]) > 1e-6 * max(1, abs(z)):
                    u.append(z)
            return len(u)
        for sig in sels:
            kinks = [bp[i] for i in range(len(bp)) if sig[i] != sig[i + 1]]
            if not kinks or abs(min(kinks) + 3) > 1e-9:
                continue
            c1 = gcount(sig, -1e5) - gcount(sig, 1e5) == 2
            c2 = gcount(sig, -a) == 3
            c3 = any(fv(sig, tt) + fv(sig, -tt) < -1e-9 for tt in np.linspace(1e-3, a - 1e-3, 400))
            if c1 and c2 and c3:
                s = fv(sig, -2 * a) + fv(sig, -a / 2) + fv(sig, 2 * a)
                found.setdefault(round(a, 6), set()).add(round(s, 6))
    exp2 = float(sp.N(sym(md_answer(1, "2"))))
    allv = {s for v in found.values() for s in v}
    sub(len(allv) == 1 and abs(list(allv)[0] - exp2) < 1e-4, f"논제2 (a→합)={found}")
    # 논제3: a_n 직접 수치적분 후 점화식·급수 확인
    def an(nn):
        zs = [(mp.pi * k) ** 4 for k in range(1, nn + 1)]
        return mp.quad(lambda xx: mp.sin(xx ** mp.mpf("0.25")) / mp.sqrt(xx), zs) / (4 * mp.pi)
    mp.mp.dps = 20
    vals = {k: an(k) for k in range(1, 7)}
    rec = all(abs(vals[k] - (-k * (-1) ** k - 1)) < 1e-8 for k in vals)
    tot = sum(mp.mpf(2) ** (vals[2 * j] - vals[2 * j - 1]) for j in range(1, 4))
    tail = sum(mp.mpf(2) ** (1 - 4 * k) for k in range(4, 80))  # a_n=-n(-1)^n-1 (위에서 확인한 점화식)로 연장
    s = tot + tail
    sub(rec and abs(s - sp.Rational(md_answer(1, "3"))) < 1e-8, f"논제3 합≈{mp.nstr(s, 10)}")
    flush(1)


def p02():
    xs = np.linspace(0, 1, 2)
    res = {}
    for dd in (0.5, -0.5):
        c = 9 / 4
        f = lambda z, d=dd: c * (1 - z**2) + d * z**3
        # 조건 확인: f(-1)+f(1)=0, f''<0, 도형 넓이 3
        zz = np.linspace(-1, 1, 200001)
        area = np.trapezoid(f(zz) - dd * zz, zz)
        assert abs(f(-1) + f(1)) < 1e-12 and abs(area - 3) < 1e-6
        fp = lambda z, d=dd: -2 * c * z + 3 * d * z**2
        assert np.all(-2 * c + 6 * dd * zz < 0)
        for n in range(1, 9):
            x_ = np.linspace(1e-9, n * np.pi - 1e-9, 4000001)
            gx = np.sin(2 * x_)
            pos = fp(gx) * 2 * np.cos(2 * x_) > 0
            res.setdefault(dd, {})[n] = pos.sum() * (x_[1] - x_[0])
    ok1 = all(abs(res[d][5] - 5 * np.pi / 2) < 2e-3 and all(abs(res[d][n] - 5 * np.pi / 2) > 0.5 for n in res[d] if n != 5) for d in res)
    sub(ok1 and int(md_answer(2, "1")) == 5, f"논제1 측도(n=5)={[round(res[d][5] / np.pi, 4) for d in res]}π")
    vals = []
    for dd in (0.5, -0.5, 0.0):
        c = mp.mpf(9) / 4
        f = lambda z, d=dd: c * (1 - z**2) + d * z**3
        hf = lambda z, d=dd: c * ((z + 1) - (z**3 + 1) / 3) + d * (z**4 - 1) / 4
        I = mp.quad(lambda z: f(z) * mp.cos(z) + hf(z) * mp.sin(z) * mp.log(mp.cos(z)) - f(z) * mp.cos(z) * mp.log(mp.cos(z)), [-1, 0, 1])
        # h 는 f 의 부정적분이므로 수치 미분으로도 확인
        assert abs(mp.diff(hf, mp.mpf("0.3")) - f(mp.mpf("0.3"))) < 1e-10
        vals.append(I)
    exp = sym(md_answer(2, "2")).evalf(30)
    sub(all(abs(v - exp) < 1e-9 for v in vals), "논제2 수치적분=" + ", ".join(mp.nstr(v, 10) for v in vals))
    flush(2)


if __name__ == "__main__":
    main(globals())
