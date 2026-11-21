#!/usr/bin/env python3
"""SJU-M26 (세종대 2026학년도 모의논술 자연계열) — sympy/수치 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-M26"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {k.strip(): sp.sympify(v.strip()) for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) g(x)=|x-1|^p 가 g(x)=8g((x+1)/2) 를 만족하는 p: x=3 에서 2^p = 8 * 1 → 풀기
    p = sp.symbols("p", real=True)
    ps = sp.solve(sp.Eq(2**p, 8 * 1), p)
    pv = ps[0]
    g = lambda v: abs(v - 1)**float(pv)
    ok1 = all(abs(g(v) - 8 * g((v + 1) / 2)) < 1e-9 * (1 + g(v)) for v in np.linspace(-7, 9, 41)) and \
        sp.sympify(md_answer(1, "1")) == pv
    # (2) 조건을 만족하는 연속함수 두 가지(비상수 φ 포함)로 ∫_2^5 / ∫_2^3 비율 계산
    A = sp.symbols("A")
    fams = [lambda v: 5 * abs(v - 1)**3,
            lambda v: abs(v - 1)**3 * (2 + mp.sin(2 * mp.pi * mp.log(abs(v - 1), 2)))]
    rat = []
    for f in fams:
        assert abs(f(4.3) - 8 * f((4.3 + 1) / 2)) < 1e-9 * f(4.3)
        rat.append(mp.quad(f, [2, 3, 5]) / mp.quad(f, [2, 3]))
    md2 = sp.sympify(md_answer(1, "2"), locals={"A": A})
    ok2 = all(abs(r - float(md2.subs(A, 1))) < 1e-9 for r in rat)
    # (3) f=c g 만이 h 연속을 허용: h(x)=h(1+(x-1)/2^n) 수치 확인 + md f 가 조건·h(1)=5 연속을 만족
    fx = sp.sympify(md_answer(1, "3").replace("|", "Abs"), locals={"x": x, "Abs": sp.Abs})
    fl = sp.lambdify(x, fx)
    ok3 = all(abs(fl(v) - 8 * fl((v + 1) / 2)) < 1e-9 * (1 + abs(fl(v))) for v in np.linspace(-5, 7, 37))
    h = lambda v: fl(v) / g(v)
    ok3 &= all(abs(h(1 + d) - 5) < 1e-9 for d in (1e-3, -1e-4, 2.0, -3.0))
    ok3 &= all(abs((v / 2**n + 1 / 2**n + sum(1 / 2**k for k in range(1, n))) - (1 + (v - 1) / 2**n)) < 1e-12
               for v in (-3.0, 0.5, 7.0) for n in range(1, 30))
    report(1, ok1 and ok2 and ok3, f"p={pv} ratio={float(rat[0]):.6f},{float(rat[1]):.6f} f=5|x-1|^3 조건 OK [증명-수치확인]")


def p02():
    r3 = mp.sqrt(3) / 2
    f = lambda s: abs(s - r3) - r3
    g = lambda v: f(mp.sin(v))
    I1 = mp.quad(g, [-mp.pi, 0])
    I2 = mp.quad(g, [0, mp.pi / 3, 2 * mp.pi / 3, mp.pi])
    md1 = parts(md_answer(2, "1"))
    ok1 = abs(I1 - float(md1["I1"])) < 1e-12 and abs(I2 - float(md1["I2"])) < 1e-12
    # (2) α 격자에서 (-π,π) 안의 극솟점 존재 여부를 직접 탐지
    xs = np.linspace(-np.pi, np.pi, 200001)[1:-1]
    def has_min(al):
        hv = np.abs(np.sin(al * xs) - np.sqrt(3) / 2) - np.sqrt(3) / 2
        i = np.arange(1, len(hv) - 1)
        return bool(np.any((hv[i] < hv[i - 1] - 1e-15) & (hv[i] <= hv[i + 1])))
    als = np.arange(0.01, 1.0, 0.0005)
    good = [a for a in als if not has_min(a)]
    amax = max(good)
    ok2 = abs(amax - float(sp.sympify(md_answer(2, "2")))) < 1e-3 and all(not has_min(a) for a in als if a <= amax)
    # (3) F(x)=∫_0^x f(sin6t)dt 를 미세 격자 누적 적분으로
    ts = np.linspace(0, np.pi, 1200001)
    fv = np.abs(np.sin(6 * ts) - np.sqrt(3) / 2) - np.sqrt(3) / 2
    F = np.concatenate([[0], np.cumsum((fv[1:] + fv[:-1]) / 2 * np.diff(ts))])
    md3 = parts(md_answer(2, "3"))
    ok3 = abs(F.min() - float(md3["min"])) < 1e-8 and abs(F.max() - float(md3["max"])) < 1e-8
    report(2, ok1 and ok2 and ok3, f"I1={float(I1):.6f} I2={float(I2):.6f} alpha_max~{amax:.4f} F[min,max]=[{F.min():.8f},{F.max():.8f}]")


def p03():
    L2, L3 = mp.log(2), mp.log(3)
    w = lambda t: mp.e**(-t) * mp.log(t)
    f = lambda v: mp.quad(lambda t: abs(v - t) * w(t), sorted({L2, min(max(v, L2), L3), L3}))
    g = lambda v: mp.quad(w, [L2, v])
    v1 = f(1)
    ok1 = abs(v1 - float(sp.sympify(md_answer(3, "1")))) < 1e-12
    # (2) [증명-수치확인] ln x <= x-1 격자, g(ln3) 부호와 상계
    ok2 = all(mp.log(v) <= v - 1 + 1e-15 for v in np.linspace(0.01, 50, 5000))
    gl3 = g(L3)
    bound = mp.quad(lambda t: mp.e**(-t) * (t - 1), [L2, L3])
    ok2 &= gl3 < 0 and gl3 <= bound < 0
    # (3) f 를 직접 최대화 (격자 → 황금분할 정밀화)
    grid = [L2 + (L3 - L2) * i / 400 for i in range(401)]
    a0 = max(grid, key=f)
    lo, hi = a0 - (L3 - L2) / 400, a0 + (L3 - L2) / 400
    phi = (mp.sqrt(5) - 1) / 2
    for _ in range(80):
        m1, m2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if f(m1) > f(m2):
            hi = m2
        else:
            lo = m1
    a = (lo + hi) / 2
    ratio = g(a) / gl3
    ok3 = abs(ratio - float(sp.sympify(md_answer(3, "3")))) < 1e-8
    report(3, ok1 and ok2 and ok3, f"f(1)={float(v1):.12f} g(ln3)={float(gl3):.6f} a={float(a):.10f} ratio={float(ratio):.10f}")


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
