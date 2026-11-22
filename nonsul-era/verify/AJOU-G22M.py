#!/usr/bin/env python3
"""AJOU-G22M (아주대 2022 실전 의학과 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G22M"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(no: int, sub: str) -> dict:
    """'a=..; b=..' 꼴 answer.value 를 사전으로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

import mpmath as mp


def p01():
    nn, ll, kk = sp.symbols("n l k", positive=True, integer=True)
    loc = {"n": nn, "l": ll, "k": kk, "E": sp.E}
    mdE = sp.sympify(md_answer(1, "1-1"), locals=loc)
    rel12 = sp.sympify(md_answer(1, "1-2"), locals=loc)
    rel22 = sp.sympify(md_answer(1, "2-2"), locals=loc)
    L_md = kv(1, "2-1")
    even = sp.sympify(L_md["even"], locals=loc)
    odd = sp.sympify(L_md["odd"], locals=loc)
    kexpr = sp.sympify(L_md["k"], locals=loc)
    ok1 = ok2 = ok3 = ok4 = True
    ks = set()
    mp.mp.dps = 30
    for a in range(1, 7):
        for b in range(1, 7):
            g = sp.sin(sp.Rational(a, b) * t)
            E = sp.integrate(sp.diff(g, t)**2, (t, 0, 2 * sp.pi))
            ok1 = ok1 and sp.simplify(E - mdE.subs({nn: a, ll: b})) == 0
            ok2 = ok2 and (bool(E >= sp.pi) == bool(rel12.subs({nn: a, ll: b})))
            # 총 이동거리: ∫ sqrt(1+f'(x(t))^2) |x'(t)| dt (f = cosh), x'(t) 부호 바뀌는 점에서 끊어 적분
            r = mp.mpf(a) / b
            brk = [mp.mpf(0)] + [(mp.pi / 2 + j * mp.pi) / r for j in range(0, 30) if (mp.pi / 2 + j * mp.pi) / r < 2 * mp.pi] + [2 * mp.pi]
            Lnum = mp.quad(lambda tt: mp.sqrt(1 + mp.sinh(mp.sin(r * tt))**2) * abs(r * mp.cos(r * tt)), brk)
            k = int(kexpr.subs({nn: a, ll: b}))
            ks.add(k)
            Lmd = (even if k % 2 == 0 else odd).subs({nn: a, ll: b, kk: k})
            ok3 = ok3 and abs(Lnum - mp.mpf(sp.N(Lmd, 30))) < mp.mpf("1e-20")
            ok4 = ok4 and ((Lnum >= mp.e - 1 / mp.e - mp.mpf("1e-20")) == bool(rel22.subs({nn: a, ll: b})))
    ok5 = sorted(ks) == [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]   # 원 해설의 k 목록과 대조
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"E 식 36쌍 일치={ok1}, E>=π 집합={ok2}, L 수치적분 36쌍 일치={ok3}, L>=e-1/e 집합={ok4}, k값={sorted(ks)}")


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
