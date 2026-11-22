#!/usr/bin/env python3
"""SMWU-M24 (숙명여대 2024학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "SMWU-M24"
WIDTH = 2
RESULTS = []
x, c = sp.symbols("x c", real=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # [증명-수치확인] (1-1) n=3..300 에서 (n+1)ln n > n ln(n+1), 귀납 단계의 핵심 사실 (k+1)^2/k > k+2
    k = sp.symbols("k", positive=True)
    ok_base = sp.log(81) > sp.log(64)
    ok_n = all(sp.N((n + 1) * sp.log(n) - n * sp.log(n + 1), 30) > 0 for n in range(3, 301))
    ok_n2_fails = sp.N(3 * sp.log(2) - 2 * sp.log(3)) < 0            # n=2 에서는 거짓 → 출발점 n=3 이 필요
    ok_key = sp.simplify((k + 1)**2 / k - (k + 2) - 1 / k) == 0
    # 귀납 단계 항등식: (k+2)ln(k+1) - (k+1)ln k + k ln(k+1) = (k+1) ln((k+1)^2/k)
    lhs = (k + 2) * sp.log(k + 1) - (k + 1) * sp.log(k) + k * sp.log(k + 1)
    ok_id = sp.simplify(sp.expand_log(lhs - (k + 1) * sp.log((k + 1)**2 / k), force=True)) == 0
    # (1-2) f=ln x/x, f'=(1-ln x)/x^2 < 0 (x>=3), 격자 1000점에서 f(x)>f(x+1)
    f = sp.log(x) / x
    fp = sp.simplify(sp.diff(f, x))
    ok_fp = sp.simplify(fp - (1 - sp.log(x)) / x**2) == 0
    grid = [3 + i * 0.1 for i in range(1000)]
    fl, fpl = sp.lambdify(x, f), sp.lambdify(x, fp)
    ok_dec = all(fpl(g) < 0 and fl(g) > fl(g + 1) for g in grid) and float(sp.E) < 3
    ok_md = md_answer(1, "1") is None and md_answer(1, "2") is None
    report(1, all([ok_base, ok_n, ok_n2_fails, ok_key, ok_id, ok_fp, ok_dec, ok_md]),
           "[증명-수치확인] n=3..300 부등식, 귀납 항등식, f'<0·f(x)>f(x+1) on [3,103]")


def p02():
    t = sp.symbols("t", real=True)
    g = -4 * x**3 + 6 * x**2 + 24 * x + 5
    # ① 에서 f 를 직접 구함: f(x) = ∫_0^x (g'+g)dt - x g(x) - c
    f = sp.expand(sp.integrate((sp.diff(g, x) + g).subs(x, t), (t, 0, x)) - x * g - c)
    h = f + c
    crit = sorted(sp.solve(sp.diff(h, x), x))
    # (2-1) 서로 다른 실근 3개인 c: 후보 = 극값들 + 각 구간 대표값 → 실근 개수 직접 셈
    vals = sorted(set(h.subs(x, r) for r in crit))
    cands = vals + [vals[0] - 1] + [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] + [vals[-1] + 1]
    def nroots(cv):
        return len(set(sp.Poly(f.subs(c, cv), x).real_roots()))
    three = {cv for cv in cands if nroots(cv) == 3}
    ok1 = three == set(sp.sympify(md_answer(2, "1")))
    # (2-2) -19<c<8: |f| 의 서로 다른 극값 = {0} ∪ {|f(crit)|}; 개수가 3 인 c 를 방정식으로 찾고 확인
    lo, hi = vals[0], sorted(vals)[1]
    absvals = [sp.Abs(f.subs(x, r)) for r in crit]
    sols = set()
    for A, B in itertools.combinations(absvals, 2):
        for s in sp.solve(sp.Eq(A, B), c):
            if lo < s < hi:
                sols.add(s)
    def ndist(cv):
        return len({sp.nsimplify(abs(f.subs({x: r, c: cv}))) for r in crit} | {0})
    sols = {s for s in sols if ndist(s) == 3}
    ok_generic = ndist(sp.Rational(1, 7)) == 4
    ok2 = sols == set(sp.sympify(md_answer(2, "2"))) and ok_generic
    report(2, ok1 and ok2, f"f={f}, crit vals={vals}, (2-1) c={sorted(three)}, (2-2) c={sorted(sols)}")


def p03():
    # (3-1) 주머니 k 에서 2^(k-1)개 → 부족분 = Σ_{k∈S} 2^(k-1); 전수 탐색
    target = 2**4 + 2**5 + 2**8
    pouches = [set(S) for r in range(1, 11) for S in itertools.combinations(range(1, 11), r)
               if sum(2**(k - 1) for k in S) == target]
    ans1 = md_answer(3, "1").replace(" ", "")
    parts = dict(seg.split("=", 1) for seg in ans1.split(";"))
    md_p = set(sp.sympify(parts["pouches"]))
    md_A = sorted(int(v) for v in sp.sympify(parts["A10"]))
    def distinct_sums(A):
        sums = [sum(S) for r in range(1, len(A) + 1) for S in itertools.combinations(A, r)]
        return len(set(sums)) == len(sums), len(set(sums))
    dA, nA = distinct_sums(md_A)
    ok1 = len(pouches) == 1 and pouches[0] == md_p and len(md_A) == 10 and dA and sum(md_A) == 2**10 - 1
    # (3-2) |C| = 2^10-1 (부분집합 합이 모두 다를 때), 최대 원소 104 이면 합 ≤ 10*104-45 < 1023
    ans2 = dict(seg.split("=", 1) for seg in md_answer(3, "2").replace(" ", "").split(";"))
    N = 2**10 - 1
    xmin = sp.ceiling(sp.Rational(2**10 + 44, 10))
    exists = (10 * 104 - 45) >= N
    # 소규모 확인: 원소 4개, 부분집합 합 서로 다름 → 합 ≥ 2^4-1 (전수, 원소 ≤ 20)
    small = min(sum(A) for A in itertools.combinations(range(1, 21), 4) if distinct_sums(list(A))[0])
    ok2 = sp.sympify(ans2["count"]) == N == nA and (ans2["exists"] == "True") == exists and xmin == 107 and small >= 2**4 - 1
    report(3, ok1 and ok2, f"pouches={pouches}, |C|={nA}, x_min={xmin}, 10*104-45={10*104-45}<{N}, small-case min sum={small}")


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
