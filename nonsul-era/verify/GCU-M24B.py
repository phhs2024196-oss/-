#!/usr/bin/env python3
"""가천대 2024학년도 논술 모의고사 B형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = "GCU-M24B"
WIDTH = 2
RESULTS = []
x, t, a, b, k, n, r = sp.symbols("x t a b k n r", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int) -> str:
    """YAML 의 answer.value (문자열)."""
    it = yaml.safe_load((ROOT / "items/GCU/PAST" / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def num(s: str):
    return sp.nsimplify(sp.sympify(s))


def nums(s: str):
    """답 문자열에서 동그라미 번호를 뺀 수들을 차례로."""
    s = re.sub(r"[①②③④]", " ", s)
    return [sp.Rational(v) for v in re.findall(r"-?\d+(?:/\d+)?", s)]


def p05():
    good = []
    for m in range(1, 300):
        cnt = 0
        for xv in range(-20, 21):
            e = xv**2 - xv * sp.log(sp.sqrt(3) * m, 3) + sp.log(sp.sqrt(m), 3)
            v = sp.N(e, 60)
            if v <= sp.Float("1e-40"):
                cnt += 1
        if cnt == 1:
            good.append(m)
    report(5, len(good) == num(ans(5)), f"정수해 1개인 n={good} (n<300 전수, 60자리 수치)")


def p06():
    f = (x**3 + a * x + b) / (x - 1)                          # x≠1,2
    s = sp.solve([sp.limit(f, x, 2) - 1, (x**3 + a * x + b).subs(x, 1)], [a, b], dict=True)[0]
    val = sp.limit(f.subs(s), x, 1)
    report(6, val == num(ans(6)), f"{s}, f(1)={val}")


def p07():
    A1, R = sp.symbols("A1 R", real=True)
    sols = sp.solve([A1**2 * R**5 - sp.Rational(5, 4), A1**2 * R**23 - 20], [A1, R], dict=True)
    vals = {sp.simplify((A1**2 * R**14).subs(s)) for s in sols if all(v.is_real for v in s.values())}
    got = nums(ans(7))
    # ① a8 ② a12 : 첨자 등비중항 3+13=2·8, 4+12=2·8
    report(7, vals == {got[-1]} and got[-2] == sp.Rational(5, 4) * 20 and "a_8" in ans(7) and "a_{12}" in ans(7),
           f"실수해 {len(sols)}개, a8^2={vals}, a8^4={sp.Rational(5, 4) * 20}")


def p08():
    f = sp.diff(x**3 + a * x + b, x)
    av = sp.solve(sp.Eq(f.subs(x, -1), 1), a)[0]
    bv = sp.solve(sp.Eq((x**3 + a * x + b).subs({a: av, x: 1}), 0), b)[0]
    val = sp.integrate(f.subs(a, av), (x, av, bv))
    report(8, val == num(ans(8)), f"f={f.subs(a, av)}, a={av}, b={bv}, ∫={val}")


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
