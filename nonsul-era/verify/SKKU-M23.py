#!/usr/bin/env python3
"""SKKU-M23 (성균관대 2023학년도 모의논술 자연계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산(전수 열거 포함).
p<번호>() 함수를 추가하면 자동 실행된다."""
import ast
import itertools
import re
import sys

import sympy as sp

PREFIX = "SKKU-M23"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    al, be = sp.symbols("alpha beta")
    fn = lambda n: sum(al**(n - k) * be**k for k in range(n + 1))
    a_, b_ = -(al + be), -al * be                                  # x^2+ax-b=0 의 두 근
    ok1 = all(sp.expand(fn(n + 2) - (-a_ * fn(n + 1) + b_ * fn(n))) == 0 for n in range(1, 12))
    # (ii) 정수 a,b 에 대해 f_n 을 근으로 직접 계산해 정수인지 [증명-수치확인]
    x = sp.symbols("x")
    ok2 = True
    for A in range(-4, 5):
        for B in range(-4, 5):
            r = sp.solve(x**2 + A * x - B, x)
            r1, r2 = (r[0], r[-1])
            for n in range(1, 9):
                v = sp.nsimplify(sp.expand(sum(r1**(n - k) * r2**k for k in range(n + 1))))
                ok2 = ok2 and v.is_integer
    # (iii) 5번 던진 모든 앞뒤 결과(32가지)를 직접 열거, f5 는 근으로 직접 계산
    cnt = 0
    for seq in itertools.product("HT", repeat=5):
        A, B = seq.count("H"), seq.count("T")
        r = sp.solve(x**2 + A * x - B, x)
        r1, r2 = r[0], r[-1]
        f5 = sp.nsimplify(sp.expand(sum(r1**(5 - k) * r2**k for k in range(6))))
        cnt += int(bool(abs(f5) > 1000))
    ok3 = cnt == int(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"recurrence ok={ok1} integer ok={ok2} count={cnt}")


def p02():
    # (i) S 를 전수로 구성: 0<=a,b<=400 범위의 15a-25b-2 가 3의 배수이고 음이 아닌 x
    S = set()
    for a in range(0, 400):
        for b in range(0, 400):
            v = 15 * a - 25 * b - 2
            if v >= 0 and v % 3 == 0 and v // 3 <= 1000:
                S.add(v // 3)
    seq = sorted(S)
    ok1 = seq[99] == int(md_answer(2, "1")) and seq[:200] == sorted(S)[:200]
    ok2 = sum(seq[n - 1]**2 for n in range(11, 21)) == int(md_answer(2, "2"))
    # (iii) T_n:T_{2n} 가 n 에 무관한 c 를 기호로 찾기
    c, n = sp.symbols("c n")
    k = sp.symbols("k", integer=True, positive=True)
    an = lambda m: seq[m - 1]
    # a_n 의 닫힌 식은 수치 seq 로부터 보간(등차 가정 없이 1차 다항 맞춤 후 검사)
    poly = sp.interpolate([(m, an(m)) for m in range(1, 4)], n)
    assert all(poly.subs(n, m) == an(m) for m in range(1, 200))
    bk = poly.subs(n, k) - c * (k - 1)
    T = sp.simplify(sp.summation(bk, (k, 1, n)))
    ratio = sp.simplify(T / T.subs(n, 2 * n))
    cs = sp.solve(sp.diff(sp.together(ratio), n).as_numer_denom()[0].expand().as_poly(n).coeffs(), c)
    cs = {s[0] if isinstance(s, tuple) else s for s in (cs if isinstance(cs, list) else [cs])}
    ok3 = cs == ast.literal_eval(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"a_100={seq[99]} sum={sum(seq[n_-1]**2 for n_ in range(11,21))} c={sorted(cs)}")


def p03():
    import numpy as np
    from mpmath import mp, quad
    mp.dps = 30
    f = lambda v: v - abs(v + 2) + abs(v + 1) - abs(v - 1) + abs(v - 2)
    g = lambda u: abs(u)**3 - u**2
    h = lambda v: g(f(v))
    xs = np.round(np.linspace(-6, 6, 120001), 10)          # 간격 1e-4 (분수점 2/3 등 근처 포함)
    def extrema(F):
        v = np.array([F(t) for t in xs])
        mx = [xs[i] for i in range(1, len(xs) - 1) if v[i] >= v[i - 1] and v[i] >= v[i + 1] and (v[i] > v[i - 1] or v[i] > v[i + 1])]
        mn = [xs[i] for i in range(1, len(xs) - 1) if v[i] <= v[i - 1] and v[i] <= v[i + 1] and (v[i] < v[i - 1] or v[i] < v[i + 1])]
        return mx, mn
    fmx, _ = extrema(f)
    ok1 = sorted(round(t, 3) for t in fmx) == sorted(float(s_) for s_ in ast.literal_eval(md_answer(3, "1")))
    hmx, hmn = extrema(h)
    parts = dict(seg.split("=") for seg in md_answer(3, "2").replace(" ", "").split(";"))
    ok2 = abs(sum(abs(t) for t in hmx) - float(sp.sympify(parts["max_abs_sum"]))) < 1e-3 and \
        abs(sum(abs(t) for t in hmn) - float(sp.sympify(parts["min_abs_sum"]))) < 1e-3 and len(hmx) == 5 and len(hmn) == 6
    I = quad(lambda t: h(t), [-2, -1, 0, 1, 2])            # 절댓값 꺾이는 점에서 분할
    ok3 = abs(I - mp.mpf(sp.Rational(md_answer(3, "3")).p) / sp.Rational(md_answer(3, "3")).q) < mp.mpf("1e-20")
    report(3, ok1 and ok2 and ok3, f"f max at {[round(float(t),3) for t in fmx]}; h max at {[round(float(t),3) for t in hmx]}; h min at {[round(float(t),4) for t in hmn]}; int={mp.nstr(I, 15)}")


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
