#!/usr/bin/env python3
"""가천대 2022학년도 논술 모의고사(인문·자연 공통) 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-M22A"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "GCU" / "PAST"
x, t, a, b, k, m = sp.symbols("x t a b k m", real=True)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def yans(seq):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return d["subparts"][0]["answer"]["value"]


def nums(s):
    """YAML answer 문자열에서 수(분수 포함)를 sympy 로 읽는다."""
    s = s.replace("\\log_2 3", "log(3,2)")
    return [sp.sympify(v) for v in re.findall(r"-?\d+/\d+|-?\d+|log\(3,2\)", s)]


def p05():
    K, y = sp.symbols("K y", real=True)
    f = 2**(x - 1) + K
    ginv = sp.solve(sp.Eq(y, f), x)[0].subs(y, x)          # 역함수 g(x)
    Ks = sp.solve(sp.Eq(ginv.subs(x, 5), 2), K)
    g35 = sp.simplify(ginv.subs({K: Ks[0], x: 35}))
    ya = yans(5)
    ok = sp.simplify(ginv - (sp.log(x - K, 2) + 1)) == 0 and Ks == [3] and g35 == 6 and "② 3" in ya and "③ 6" in ya
    report(5, ok, f"g(x)={sp.simplify(ginv)}, k={Ks}, g(35)={g35}")


def p06():
    th = sp.symbols("theta", real=True)
    D = 3*sp.sin(th)**2 - 4*(sp.cos(th) - sp.Rational(1, 4))
    S = sp.solveset(D >= 0, th, sp.Interval.Ropen(0, 2*sp.pi))
    al, be = S.inf, S.sup
    val = sp.simplify(sp.tan(al) - sp.tan(be))
    ok = isinstance(S, sp.Interval) and sp.simplify(sp.cos(al) - sp.Rational(2, 3)) == 0 and sp.simplify(val - sp.sqrt(5)) == 0 \
        and yans(6).strip() == "√5"
    report(6, ok, f"해집합={S}, tanα-tanβ={val}")


def p07():
    found = []
    tot = sp.Integer(0)
    for M in range(1, 40):
        r1, r2 = sp.solve(x**2 + 25*x - (2*M - 1)*(2*M + 1), x)
        tot += sp.radsimp(1/r1 + 1/r2)
        if tot == 12:
            found.append(M)
    # 닫힌 꼴 25m/(2m+1) 은 m 에 대해 증가하고 25/2 미만 → m<40 전수로 충분
    ok = found == [int(yans(7))]
    report(7, ok, f"두 근으로 직접 합산(m<40 전수), 합=12 인 m={found}")


def p08():
    c = sp.symbols("c")
    f = sp.integrate(3*x**2 - 2*x + 1, x) + c
    tang = sp.diff(f, x).subs(x, 1)*(x - 1) + f.subs(x, 1)
    cs = sp.solve(tang.subs(x, -1), c)
    F = f.subs(c, cs[0])
    yf = sp.sympify(yans(8).split("=")[1].replace("^", "**"), locals={"x": x})
    ok = sp.expand(F - yf) == 0 and sp.diff(F, x) == 3*x**2 - 2*x + 1
    report(8, ok, f"c={cs}, f(x)={F}")


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
