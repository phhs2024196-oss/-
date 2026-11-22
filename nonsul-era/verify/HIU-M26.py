#!/usr/bin/env python3
"""HIU-M26 (홍익대 2026학년도 모의논술 자연) — sympy 검산.
규칙: 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
증명 소문항 (2)(4)는 기호 단순화 + 수치 표본으로 확인 [증명-수치확인]."""
import re
import sys

import sympy as sp

PREFIX = "HIU-M26"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def parts(s: str) -> dict:
    """'a = 1; b = 2' 형식 → {'a': '1', 'b': '2'}"""
    return {kk.strip(): v.strip() for kk, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def position(nbars, tt):
    """k번째 막대: 앞 막대 기준 1초에 1바퀴 → 테이블 기준 각은 상대 회전각의 합 = k*2*pi*t, 길이 1/k."""
    X = Y = 0
    for kk in range(1, nbars + 1):
        ang = sum(2 * sp.pi * tt for _ in range(kk))          # 상대 회전 kk 개 누적
        X += sp.cos(ang) / kk
        Y += sp.sin(ang) / kk
    return X, Y


def p01():
    import random
    random.seed(1)
    al, be = sp.symbols("alpha beta", real=True)
    # (1)
    f, g = position(2, t)
    a1 = parts(md_answer(1, "1"))
    ok1 = sp.simplify(f - sp.sympify(a1["f"], locals={"t": t})) == 0 and \
        sp.simplify(g - sp.sympify(a1["g"], locals={"t": t})) == 0 and \
        position(2, 0) == (1 + sp.Rational(1, 2), 0)
    # (2) [증명-수치확인] 배각 등식과 f'^2+g'^2 = 16 pi^2 cos^2(pi t)
    ok2 = sp.simplify(sp.expand_trig(sp.sin(2 * al)) - 2 * sp.sin(al) * sp.cos(al)) == 0 and \
        sp.simplify(sp.expand_trig(sp.cos(2 * al)) - (sp.cos(al) ** 2 - sp.sin(al) ** 2)) == 0
    sp2 = sp.diff(f, t) ** 2 + sp.diff(g, t) ** 2
    ok2 = ok2 and sp.simplify(sp.expand_trig(sp2 - 16 * sp.pi**2 * sp.cos(sp.pi * t) ** 2)) == 0 and \
        all(abs(float((sp2 - 16 * sp.pi**2 * sp.cos(sp.pi * t) ** 2).subs(t, random.uniform(-3, 3)))) < 1e-9 for _ in range(20))
    # (3) 거리: 속력 sqrt(f'^2+g'^2) 를 [0,1] 에서 수치 적분
    import mpmath as mp
    fl = sp.lambdify(t, sp.sqrt(sp2), "mpmath")
    dist = mp.quad(fl, [0, 0.5, 1])
    ok3 = abs(dist - float(sp.sympify(md_answer(1, "3")))) < 1e-9
    # (4) [증명-수치확인] 곱→차 등식과 두 합 공식 (n=1..40, 무작위 x 20개)
    ok4 = sp.simplify(sp.expand_trig(sp.cos(al - be) - sp.cos(al + be)) - 2 * sp.sin(al) * sp.sin(be)) == 0 and \
        sp.simplify(sp.expand_trig(sp.sin(al - be) - sp.sin(al + be)) + 2 * sp.cos(al) * sp.sin(be)) == 0
    for _ in range(20):
        xv = random.uniform(-7, 7)
        for nn in range(1, 41):
            l1 = sum(mp.sin(kk * xv) for kk in range(1, nn + 1)) * mp.sin(xv / 2)
            r1 = mp.sin((nn + 1) * xv / 2) * mp.sin(nn * xv / 2)
            l2 = sum(mp.cos(kk * xv) for kk in range(1, nn + 1)) * mp.sin(xv / 2)
            r2 = mp.cos((nn + 1) * xv / 2) * mp.sin(nn * xv / 2)
            ok4 = ok4 and abs(l1 - r1) < 1e-9 and abs(l2 - r2) < 1e-9
    # (5) P_100 속력: 직접 합으로 계산 → YAML 식과 비교, 영점 개수 독립 세기
    a5 = md_answer(1, "5")
    m = re.search(r"speed\s*=\s*(.+?)\s*\(t not integer\),\s*(.+?)\s*\(t integer\);\s*count\s*=\s*(\d+)", a5)
    md_speed = sp.lambdify(t, sp.sympify(m.group(1), locals={"t": t}), "mpmath")
    md_int = float(sp.sympify(m.group(2)))
    md_cnt = int(m.group(3))
    def speed(tv):
        fx = -2 * mp.pi * sum(mp.sin(2 * mp.pi * kk * tv) for kk in range(1, 101))
        gy = 2 * mp.pi * sum(mp.cos(2 * mp.pi * kk * tv) for kk in range(1, 101))
        return mp.sqrt(fx**2 + gy**2)
    ok5 = abs(speed(1) - md_int) < 1e-6 and abs(speed(2) - md_int) < 1e-6
    for _ in range(20):
        tv = random.uniform(0.001, 0.999)
        ok5 = ok5 and abs(speed(tv) - md_speed(tv)) < 1e-7 * max(1, abs(speed(tv)))
    # 영점: |sum e^{2pi i k t}| 의 격자 국소최솟값 중 0 근처인 것 세기 (구간 (0,1])
    import numpy as np
    ts = np.linspace(0, 1, 400001)[1:]
    ks = np.arange(1, 101)
    vals = np.empty_like(ts)
    for i in range(0, len(ts), 20000):
        blk = ts[i:i + 20000]
        vals[i:i + 20000] = np.abs(np.exp(2j * np.pi * np.outer(blk, ks)).sum(axis=1))
    loc = (vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:]) & (vals[1:-1] < 0.05)
    cnt = int(loc.sum())
    exact = all(abs(complex(sum(sp.exp(2 * sp.pi * sp.I * kk * sp.Rational(mm, 100)) for kk in range(1, 101)).evalf())) < 1e-9
                for mm in (1, 37, 99))
    ok5 = ok5 and cnt == md_cnt and exact
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"(1)ok={ok1} (2)ok={ok2} dist={float(dist):.10f} (4)ok={ok4} speed(1)={float(speed(1)):.6f} zeros={cnt}")


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
