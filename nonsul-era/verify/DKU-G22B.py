"""DKU-G22B 검산: 문제 조건에서 독립 재계산한 값을 items/DKU/PAST 의 answer.value 와 비교."""
PREFIX = "DKU-G22B"
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


def p01():
    mp.mp.dps = 60
    # 논제1: 역함수 존재 => f'(x)=1/x+4x+a >= 0 (x>0) => a >= -4. L 이 0 이 아니려면 f'(f^{-1}(1))=0.
    def Lval(a, b, d):
        f = lambda z: mp.log(z) + 2 * z**2 + a * z + b
        fp = lambda z: 1 / z + 4 * z + a
        xv = 1 + d
        # f^{-1}(xv), f^{-1}(1) 을 이분법으로
        def inv(y):
            lo, hi = mp.mpf("1e-6"), mp.mpf(10)
            for _ in range(200):
                mid = (lo + hi) / 2
                if f(mid) < y:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        yv, p = inv(xv), inv(1)
        return (1 / fp(yv)) * mp.log(yv / p) ** 2
    a0 = -4
    b0 = sp.Rational(5, 2) + sp.log(2)
    L1 = Lval(mp.mpf(a0), mp.mpf(sp.N(b0, 60)), mp.mpf("1e-15"))
    L2 = Lval(mp.mpf(a0), mp.mpf(sp.N(b0, 60)) + mp.mpf("0.3"), mp.mpf("1e-15"))
    L3 = Lval(mp.mpf("-3.9"), mp.mpf("1"), mp.mpf("1e-15"))
    exp = sym(md_answer(1, "1"))
    tot = a0 + sp.N(b0, 30) + sp.Rational(1, 2)
    ok1 = abs(L1 - 0.5) < 1e-3 and abs(L2) < 1e-3 and abs(L3) < 1e-3 and abs(tot - sp.N(exp, 30)) < 1e-9
    sub(ok1, f"논제1 L={mp.nstr(L1, 6)} (b 이탈 {mp.nstr(L2, 3)}, a 이탈 {mp.nstr(L3, 3)}), a+b+L={sp.N(tot, 10)}")
    mp.mp.dps = 30
    # 논제2: g=(x-2)^m (...) 의 중복도 m=2 일 때 lim=2. g=(x-2)^2 (x-r)
    r = sp.symbols("r", real=True)
    lims = {m: sp.limit((x - 2) * sp.diff((x - 2) ** m * (x - 1), x) / ((x - 2) ** m * (x - 1)), x, 2) for m in (1, 2, 3)}
    gg = (x - 2) ** 2 * (x - r)
    hh = sp.simplify((x - 2) * sp.diff(gg, x) / gg)
    out = []
    for rv in (sp.Rational(k, 2) for k in range(-12, 13)):
        if rv == 0:
            continue
        for c in (rv, -rv):
            if c == 0:
                continue
            # h(|x|) 의 x->c 극한이 존재하지 않는 경우: |x|=r 가 되는 c=±r 이고 r>0
            if rv > 0:
                hx = sp.lambdify(x, hh.subs(r, rv), "mpmath")
                lft, rgt = hx(abs(c - mp.mpf("1e-9"))), hx(abs(c + mp.mpf("1e-9")))
                dne = abs(lft - rgt) > 1e3
            else:
                dne = False
            if dne and sp.simplify(gg.subs({r: rv, x: c}) - 18 * c) == 0:
                out.append((rv, c))
    val = [sp.simplify(gg.subs({r: rv, x: 2 * c})) for rv, c in out]
    sub(lims[2] == 2 and lims[1] == 1 and len(out) == 1 and val == [sp.Rational(md_answer(1, "2"))], f"논제2 (r,c)={out}, g(2c)={val}")
    # 논제3: k=x+sin x, F'(t)=pi cos(k^{-1}(pi t)); 부호 변화로 극대/극소를 찾는다
    def kinv(y):
        lo, hi = mp.mpf(y) - 2, mp.mpf(y) + 2
        for _ in range(100):
            mid = (lo + hi) / 2
            if mid + mp.sin(mid) < y:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    ts = [mp.mpf(i) / 200 + mp.mpf("0.0007") for i in range(1, 1600)]
    sg = [mp.cos(kinv(mp.pi * s)) for s in ts]
    mx, mn = [], []
    for i in range(len(ts) - 1):
        if sg[i] > 0 > sg[i + 1]:
            mx.append(mp.findroot(lambda s: mp.cos(kinv(mp.pi * s)), (ts[i], ts[i + 1]), solver="anderson"))
        if sg[i] < 0 < sg[i + 1]:
            mn.append(mp.findroot(lambda s: mp.cos(kinv(mp.pi * s)), (ts[i], ts[i + 1]), solver="anderson"))
    al, be = max(mx), max(mn)
    pts3 = [mp.pi * al + (mp.pi * be - mp.pi * al) * i / 100 for i in range(101)]
    val3 = mp.quad(lambda y: mp.cos(kinv(y)), pts3)
    ok3 = abs(val3 - sym(md_answer(1, "3")).evalf(30)) < 1e-8
    sub(ok3, f"논제3 alpha={mp.nstr(al, 8)}, beta={mp.nstr(be, 8)}, F(b)-F(a)={mp.nstr(val3, 10)}")
    flush(1)


def p02():
    mp.mp.dps = 30
    cen = [(0, 2), (-2, 0), (0, -2), (2, 0)]
    ang = [mp.pi / 2, mp.pi, -mp.pi / 2, mp.mpf(0)]

    def P(i, tv):
        return (cen[i][0] + mp.cos(ang[i] + tv), cen[i][1] + mp.sin(ang[i] + tv))

    def ell(tv):
        a, b = P(0, tv), P(2, tv)
        return mp.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def area(tv):
        pts = [P(i, tv) for i in range(4)]
        s = 0
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2
    a = mp.quad(lambda z: ell(z) ** 4, [0, mp.pi / 2]) - 40 * mp.quad(lambda z: ell(z) ** 2, [0, mp.pi / 2]) + 200 * mp.pi
    # 극값: A 를 float 로 빠르게 평가해 격자에서 국소 극점 탐색 후 황금분할로 정밀화
    af = float(a)

    def Af(tv):
        pts = [(cen[i][0] + np.cos(float(ang[i]) + tv), cen[i][1] + np.sin(float(ang[i]) + tv)) for i in range(4)]
        s = 0.0
        for i in range(4):
            s += pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1]
        return abs(s) / 2
    xs = np.linspace(1e-4, af, 40001)
    ys = np.array([Af(v) for v in xs])
    ext = []
    for i in range(1, len(xs) - 1):
        if (ys[i] - ys[i - 1]) * (ys[i + 1] - ys[i]) < 0:
            lo, hi = xs[i - 1], xs[i + 1]
            sgn = 1 if ys[i] > ys[i - 1] else -1
            for _ in range(80):
                m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                if sgn * Af(m1) < sgn * Af(m2):
                    lo = m1
                else:
                    hi = m2
            tt = (lo + hi) / 2
            if tt < af - 1e-6:
                ext.append((tt, Af(tt)))
    tot = sum(v for _, v in ext)
    ok1 = abs(a - 64 * mp.pi) < 1e-9 and abs(tot - int(md_answer(2, "1"))) < 1e-6
    sub(ok1, f"논제1 a/pi={mp.nstr(a / mp.pi, 10)}, 극값 {len(ext)}개 합={tot:.6f}")
    # 논제2: Q 는 S 둘레를 시계 방향. f=(x_P1+x_P3+x_Q)/3
    V = [(3 * mp.pi / 2, 0), (3 * mp.pi / 2, -mp.pi / 2), (mp.pi, -mp.pi / 2), (mp.pi, 0)]
    per = 2 * mp.pi

    def Qx(tv):
        s = tv % per
        for i in range(4):
            A_, B_ = V[i], V[(i + 1) % 4]
            Ls = abs(B_[0] - A_[0]) + abs(B_[1] - A_[1])
            if s <= Ls:
                return A_[0] + (B_[0] - A_[0]) * s / Ls
            s -= Ls
        return V[0][0]

    def fq(tv):
        return (P(0, tv)[0] + P(2, tv)[0] + Qx(tv)) / 3
    br = [mp.pi / 2, mp.pi, 3 * mp.pi / 2]
    I1 = mp.quad(lambda z: area(z) * fq(z) ** 2, [mp.pi / 2, mp.pi, 3 * mp.pi / 2])
    I2 = mp.quad(lambda z: fq(z) ** 2, [0, mp.pi / 2, mp.pi])
    val = I1 - 10 * I2
    exp = sym(md_answer(2, "2")).evalf(30)
    sub(abs(val - exp) < 1e-8, f"논제2 numeric={mp.nstr(val, 12)} 공식값={sp.N(exp, 12)}")
    flush(2)


if __name__ == "__main__":
    main(globals())
