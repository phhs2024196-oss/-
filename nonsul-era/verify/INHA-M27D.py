#!/usr/bin/env python3
"""INHA-M27D (2027학년도 인하대 자연계열 논술 모의고사, 의예과) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML 정답과 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "INHA-M27D"
WIDTH = 2
RESULTS = []
KIT = pathlib.Path(__file__).resolve().parents[1]
x, t, s = sp.symbols("x t s", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int, sub: str) -> str:
    it = yaml.safe_load((KIT / f"items/INHA/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return next(p for p in it["subparts"] if str(p["id"]) == sub)["answer"]["value"]


def tex(v: str):
    """간단한 LaTeX(\\dfrac, \\sqrt, \\pi, \\ln) → sympy."""
    v = v.replace("\\dfrac", "\\frac")
    v = v.replace("\\pi", "{pi}")
    while "\\frac" in v:
        w = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", v)
        w = re.sub(r"\\frac(\w)(\w)", r"((\1)/(\2))", w)
        w = re.sub(r"\\frac(\w)\{([^{}]*)\}", r"((\1)/(\2))", w)
        w = re.sub(r"\\frac\{([^{}]*)\}(\w)", r"((\1)/(\2))", w)
        if w == v:
            raise ValueError(f"tex 변환 실패: {v}")
        v = w
    v = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", v)
    v = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", v)
    v = v.replace("\\pi", " pi ").replace("\\ln", " log ").replace("{", "(").replace("}", ")")
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    return parse_expr(v, local_dict={"pi": sp.pi, "log": sp.log, "sqrt": sp.sqrt},
                      transformations=standard_transformations + (implicit_multiplication_application, convert_xor))




from fractions import Fraction as Fr


def p02():
    """(2-1) a_n>ln3 (2-2) lim=ln3 (2-3) Σb_k → ln3-3/2."""
    import math
    def a(n):
        return sum(Fr(1, j) for j in range(n, 3 * n))
    ok1 = all(float(a(n)) > math.log(3) for n in range(1, 201))
    ok1b = all(float(a(n)) < math.log((3 * n - 1) / (n - 1)) for n in range(2, 201))
    big = sum(1.0 / j for j in range(10**6, 3 * 10**6))
    ok2 = abs(big - float(tex(ans(2, "2")))) < 1e-6
    k = sp.symbols("k", positive=True, integer=True)
    bk = 1 / (3 * k) + 1 / (3 * k + 1) + 1 / (3 * k + 2) - 1 / k
    tele = all(sp.nsimplify(bk.subs(k, kk)) == a(kk + 1) - a(kk) for kk in range(1, 40))
    part = sum(float(bk.subs(k, kk)) for kk in range(1, 2001))
    lim = float(tex(ans(2, "3")))
    ok3 = tele and abs(part - lim) < 1e-3 and abs((float(a(2001)) - 1.5) - part) < 1e-9
    report(2, ok1 and ok1b and ok2 and ok3, f"(2-1) n≤200 [증명-수치확인] (2-2) a(10^6)={big:.8f} (2-3) 망원 확인, Σ(2000)={part:.6f} → {lim:.6f}")


def p03():
    """(3-1) b6=14 순서쌍 전수 (3-2) b20 최솟값 — 도달 가능 집합 DP."""
    import itertools
    N = 3000
    a = [None] * (N + 1)
    a[1] = a[2] = Fr(1)
    for n in range(1, N // 3 + 1):
        p = 1
        while not (3 ** (p - 1) <= n < 3 ** p):
            p += 1
        for r in range(3):
            m = 3 * n + r
            if 2 < m <= N:
                a[m] = a[n] + Fr(r + 1, 3 ** p)
    ok = lambda kk, x, y: (a[y] > a[x]) if kk % 2 else (a[y] < a[x])
    cnt = 0
    for c in itertools.combinations(range(1, 14), 5):
        b = list(c) + [14]
        if all(ok(i + 1, b[i], b[i + 1]) for i in range(5)):
            cnt += 1
    S = set(range(1, N + 1))
    for kk in range(1, 20):
        nxt, cur_min, cur_max = set(), None, None
        for y in range(1, N + 1):
            if cur_min is not None and ((kk % 2 and cur_min < a[y]) or (kk % 2 == 0 and cur_max > a[y])):
                nxt.add(y)
            if y in S:
                cur_min = a[y] if cur_min is None else min(cur_min, a[y])
                cur_max = a[y] if cur_max is None else max(cur_max, a[y])
        S = nxt
    b20 = min(S)
    ok3 = cnt == int(ans(3, "1")) and b20 == int(ans(3, "2"))
    report(3, ok3, f"(3-1) 순서쌍 {cnt}개 (3-2) b20 최소 {b20}")


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
