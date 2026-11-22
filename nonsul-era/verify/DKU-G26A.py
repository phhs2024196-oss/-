#!/usr/bin/env python3
"""DKU-G26A (단국대 2026학년도 논술 자연계열 오전) — 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "DKU-G26A"
WIDTH = 2
RESULTS = []
x, t, k, a = sp.symbols("x t k a")
NS = {"pi": sp.pi, "E": sp.E, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"DKU-G26A-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def ans_num(no, sub):
    return float(sp.N(sp.sympify(md_answer(no, sub), locals=NS)))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"DKU-G26A-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def close(u, v, tol=1e-9):
    return abs(float(u) - float(v)) < tol



def fmin_val(av):
    """f(x)=∫_0^x (t-a)e^t dt 의 극솟값: f'=(x-a)e^x=0 의 근을 수치로 찾아 f 값을 구한다."""
    F = lambda xx: mp.quad(lambda s: (s - av) * mp.e**s, [0, xx])
    xm = mp.findroot(lambda xx: (xx - av) * mp.e**xx, av + 0.1, tol=1e-15, verify=False)      # F'(x)=(x-a)e^x (미적분 기본정리)
    # 극소 확인: 좌우에서 f 가 더 크다
    return F(xm), xm, F(xm - 0.1) > F(xm) < F(xm + 0.1)


def p01():
    mp.mp.dps = 25
    # 1. 극솟값 = 3 - e^a 를 만족시키는 a
    sol = mp.findroot(lambda av: fmin_val(av)[0] - (3 - mp.e**av), (1.5, 2.5), solver="anderson", tol=1e-18)
    ok1 = close(sol, ans_num(1, "1"), 1e-9) and fmin_val(sol)[2]
    # 2. g(a) 폐쇄식을 수치 극솟값과 대조한 뒤, h=g^{-1}(x≥0) 로 ∫_0^m F'(h(x))dx
    gnum = lambda av: fmin_val(av)[0]
    okg = all(abs(gnum(av) - (1 + av - mp.e**av)) < 1e-15 for av in (0.0, 0.7, 1.3, 2.0))
    m = 3 - mp.e**2
    hinv = lambda xv: mp.findroot(lambda uu: (1 + uu - mp.e**uu) - xv, 0.5 if xv > -1 else 1.5, tol=1e-18, verify=False)
    # g 는 x≥0 에서 감소, g(0)=0, g(2)=m<0 이므로 x∈[m,0] 에서 h(x)∈[0,2]
    Fp = lambda uu: uu * mp.e**uu
    val = mp.quad(lambda xv: Fp(hinv(xv)), [m, 0])
    val = abs(val)     # ∫_0^m = -∫_m^0
    val = -mp.quad(lambda xv: Fp(hinv(xv)), [m, 0])
    ok2 = okg and close(val, ans_num(1, "2"), 1e-8)
    # 3. 정삼각형: 변 2√3, 높이 3, 무게중심→꼭짓점 2. 윗부분(꼭짓점 y=c>0)은 높이 c 인 닮은 정삼각형
    expr = md_answer(1, "3").split(";")[0].split("=")[1]
    kf = sp.lambdify(x, sp.sympify(expr, locals=NS), "numpy")
    xs = np.linspace(-1, 1, 200001)
    gx = 1 + xs - np.exp(xs)
    c = kf(xs) + 2                                          # 위 꼭짓점 y좌표
    area = (c**2) / np.sqrt(3)                              # 높이 c, 한 변 2c/√3
    target = np.sqrt(3) / 3 * (1 + gx)
    mx = float(re.findall(r"-?\d+", md_answer(1, "3").split(";")[1])[0])
    ok3 = np.max(np.abs(area - target)) < 1e-9 and np.all(c >= 0) and abs(np.max(kf(xs)) - mx) < 1e-9
    report(1, ok1 and ok2 and ok3, f"a={mp.nstr(sol, 8)} ∫={mp.nstr(val, 10)} max k={np.max(kf(xs)):.6f}")


def p02():
    mp.mp.dps = 20
    # 1. f 가 [b,c] (b<0<c) 에서 일대일 ⇔ f' 부호가 일정(구간 안에서 증감 불변). a 가 가능한 값의 집합
    fp = lambda v: 2 * (1 - v**2) / (1 + v**2)**2

    def injective(b, c):
        xs = np.concatenate([np.linspace(b, c, 20001)[1:-1], [b + 1e-7, c - 1e-7]])
        s = np.sign(fp(xs))
        return bool(np.all(s == s[0]))

    def feasible(av, eps=1e-6):
        return injective(min(av, 0) - eps, max(av, 0) + eps)      # 가능한 (b,c) 집합은 축소에 닫혀 있다
    lo, hi = [float(v) for v in re.findall(r"-?\d+\.?\d*", md_answer(2, "1"))][:2]
    pts = list(np.linspace(-3, 3, 241)) + [-0.99999, 0.99999, -1.00001, 1.00001]
    pts = [v for v in pts if min(abs(v - 1), abs(v + 1)) > 5e-6 or abs(abs(v) - 1) > 0 ]
    ok1 = all(feasible(v) == (lo < v < hi) for v in pts if abs(abs(v) - 1) > 2e-6 or True)
    # 2. A(x) 수치 → 극값 위치, A(k)=ln2
    A = lambda xv: mp.quad(lambda s: abs(2 * s / (1 + s**2) - xv), [0, 1] if xv <= 0 or xv >= 1 else [0, mp.mpf(xv + 0) and mp.findroot(lambda z: 2 * z / (1 + z**2) - xv, 0.5), 1])
    xs = np.linspace(-1.0, 3.0, 801)
    Av = np.array([float(A(mp.mpf(v))) for v in xs])
    d = np.diff(Av)
    ext = [xs[i + 1] for i in range(len(d) - 1) if d[i] * d[i + 1] < 0]
    cval = ext[0]
    cfine = mp.findroot(lambda z: 2 * g_inv(z) - 1, 0.8) if False else mp.findroot(lambda z: (A(z + mp.mpf("1e-6")) - A(z - mp.mpf("1e-6"))) / mp.mpf("2e-6"), 0.8, tol=1e-12)
    kk = mp.findroot(lambda z: A(z) - mp.log(2), 1.4)
    c_md, k_md = [float(sp.N(sp.sympify(p_.split("=")[1], locals=NS))) for p_ in md_answer(2, "2").split(",")]
    ok2 = len(ext) == 1 and abs(cval - c_md) < 0.01 and close(cfine, c_md, 1e-6) and close(kk, k_md, 1e-9)
    report(2, ok1 and ok2, f"[{ok1},{len(ext)==1},{abs(cval - c_md) < 0.01},{close(cfine, c_md, 1e-6)},{close(kk, k_md, 1e-9)}] a 범위 ({lo},{hi}) 격자 일치, c≈{mp.nstr(cfine, 8)} (극값 {len(ext)}개), k={mp.nstr(kk, 10)}")


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
