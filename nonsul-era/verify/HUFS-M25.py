#!/usr/bin/env python3
"""한국외대 2025학년도 논술전형 모의논술(자연계) 문제 1~7 (HUFS-M25-01~07) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "HUFS-M25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/HUFS/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """f·g = x^4-6x^3+11x^2-6x 의 인수 분배를 전수 조사: f 는 최고차 1/2 삼차, g 는 (x-1) 인수, 극한 조건 확인."""
    P = x**4 - 6 * x**3 + 11 * x**2 - 6 * x
    roots = [r for r in sp.roots(sp.Poly(P, x)).keys()]
    import itertools
    sols = set()
    for fr in itertools.combinations(roots, 3):           # f 는 삼차(극한 (나))
        g_r = [r for r in roots if r not in fr]
        f = sp.Rational(1, 2) * sp.prod([x - r for r in fr])
        g = sp.expand(P / f)
        if sp.limit(f / x**3, x, sp.oo) != sp.Rational(1, 2):
            continue
        if not sp.limit(g / (x - 1), x, 1).is_finite:      # (다) 극한 존재
            continue
        if sp.simplify(g.subs(x, 1)) != 0:
            continue
        sols.add(sp.nsimplify(2 * f.subs(x, 100) / (g.subs(x, 98) * g.subs(x, 99))))
    ans = load_answer(1)
    report(1, len(sols) == 1 and list(sols)[0] == ans, f"근 {roots}, 가능한 값 {sols}, md={ans}")


def p02():
    """f 를 미정계수 삼차 이하 다항식으로 두고 항등식으로 f, a, b 결정 → ∫_{-1}^{a+b} f'."""
    c = sp.symbols("c0:4")
    f = sum(c[i] * t**i for i in range(4))
    lhs = sp.integrate((x - t) * f, (t, -1, x))
    eqs = sp.Poly(sp.expand(lhs - (x**3 + a * x**2 + b * x + 2)), x).all_coeffs()
    sol = sp.solve(eqs, list(c) + [a, b], dict=True)
    vals = set()
    for s in sol:
        fx = f.subs(s).subs(t, x)
        vals.add(sp.integrate(sp.diff(fx, x), (x, -1, s[a] + s[b])))
    ans = load_answer(2)
    report(2, len(vals) == 1 and sp.simplify(list(vals)[0] - ans) == 0,
           f"해 {sol}, 적분값 {vals}, md={ans}")


def p03():
    """C=(cosθ, sinθ) 로 매개화: b=AC, a=BC. 기호 최소(b∈[0,2]) + θ 격자 수치 최소."""
    th = sp.symbols("theta", real=True)
    B_ = sp.sqrt((sp.cos(th) + 1)**2 + sp.sin(th)**2)
    A_ = sp.sqrt((sp.cos(th) - 1)**2 + sp.sin(th)**2)
    expr = B_**4 - 2 * B_ * A_**2 + A_**2
    import math
    num = min(float(expr.subs(th, 2 * math.pi * i / 20000)) for i in range(20000))
    bb = sp.symbols("bb", real=True)
    F = bb**4 - 2 * bb * (4 - bb**2) + (4 - bb**2)
    cands = [r for r in sp.solve(sp.diff(F, bb), bb) if r.is_real and 0 <= r <= 2] + [0, 2]
    m = min(F.subs(bb, r) for r in cands)
    ans = load_answer(3)
    report(3, m == ans and abs(num - float(ans)) < 1e-6, f"임계점 {cands}, 최솟값 {m}, 격자 최소≈{num:.8f}, md={ans}")


def p04():
    """a_n 을 조건 (가)(나)로 정해 수치 합(mpmath 50자리) + 기호 망원합."""
    import mpmath as mp
    mp.mp.dps = 50
    tot = mp.mpf(0)
    for m in range(1, 100):
        cand = [mp.atan(1 / mp.sqrt(m)), mp.atan(1 / mp.sqrt(m)) + mp.pi]   # tan>0: 1·3사분면 대표
        good = [c for c in cand if mp.sin(c) + mp.cos(c) > 0]
        assert len(good) == 1
        c = good[0]
        tot += mp.sin(c) / (1 + mp.cos(c))
    sym = sp.summation(sp.sqrt(n + 1) - sp.sqrt(n), (n, 1, 99))
    ans = load_answer(4)
    report(4, abs(tot - mp.mpf(str(sp.N(ans, 40)))) < mp.mpf("1e-40") and sp.simplify(sym - ans) == 0,
           f"수치합={mp.nstr(tot, 20)}, 망원합={sym}, md={ans}")


def p05():
    """[증명-수치확인] 해설의 귀납 구성(f'=fp+gq, g'=fq+gp)으로 n=1..12 의 f,g 를 실제로 만들어 항등식 확인."""
    f = x - sp.Rational(3, 2)
    g = sp.Rational(1, 2)
    ok = sp.expand(f**2 - g**2 - (x - 1) * (x - 2)) == 0
    for kk in range(1, 12):
        p = x - (2 * kk + sp.Rational(3, 2))
        q = sp.Rational(1, 2)
        ok &= sp.expand(p**2 - q**2 - (x - (2 * kk + 1)) * (x - (2 * kk + 2))) == 0
        f, g = sp.expand(f * p + g * q), sp.expand(f * q + g * p)
        target = sp.prod([x - j for j in range(1, 2 * (kk + 1) + 1)])
        ok &= sp.expand(f**2 - g**2 - target) == 0
    ans = load_answer(5, raw=True)
    report(5, ok and ans is None, f"[증명-수치확인] n=1..12 구성 항등식 성립={ok}, answer.value={ans}")


def p06():
    """ab=2025^2 인 자연수 쌍 전수 + (가)(다)(진수 조건) 판정 → b/(3a^2)."""
    N = 2025**2
    vals = []
    for av in sp.divisors(N):
        bv = N // av
        if av <= 1:
            continue
        u, w = av**2 - 43 * av + 130, 50 - av
        if u <= 0 or w <= 0:
            continue
        # log_{√2025} √w = log_2025 w 이므로 (다) ⟺ u > w
        if sp.log(u, 2025) > sp.log(sp.sqrt(w), sp.sqrt(2025)) and u > w:
            vals.append((av, bv, sp.Rational(bv, 3 * av**2)))
    ans = load_answer(6)
    report(6, len(vals) == 1 and vals[0][2] == ans, f"후보 (a,b,값)={vals}, md={ans}")


def p07():
    """r = 넓이/반둘레 로 독립 계산 → lim_{a→0+} r/a."""
    A = sp.symbols("A", positive=True)
    PQ = sp.sqrt(A**2 + (10 - A)**2)
    QR = 2 * (10 - A)
    area = sp.Rational(1, 2) * QR * A
    r = area / ((2 * PQ + QR) / 2)
    L = sp.limit(r / A, A, 0, "+")
    # 해설 식과도 대조(0<a<10 에서 몇 점)
    rh = (-(10 - A)**2 + sp.sqrt((10 - A)**4 + A**2 * (10 - A)**2)) / A
    same = all(abs(float((r - rh).subs(A, v))) < 1e-12 for v in (0.5, 3, 7.2, 9.9))
    ans = load_answer(7)
    report(7, sp.simplify(L - ans) == 0 and same, f"lim r/a={L}, 해설 r 식과 일치={same}, md={ans}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
