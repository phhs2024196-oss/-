#!/usr/bin/env python3
"""KNU-M21A (경북대 2021 논술 모의고사 자연계열Ⅰ 수학 1·2·3) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KNU-M21A"
WIDTH = 2
RESULTS = []
mp.mp.dps = 40


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    """소문항 sub(0부터)의 answer.value 를 ';' 로 나눠 sympy 식 목록으로."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return [sp.sympify(s.strip()) for s in str(v).split(";")]




def p01():
    oks, msgs = [], []
    # 1-1 [증명-수치확인]
    ok1 = True
    for f in [mp.exp, lambda s: 1 / s, lambda s: mp.sin(s) + s ** 2]:
        for x in [mp.mpf("1.5"), mp.mpf(2), mp.mpf("3.7")]:
            L = mp.quad(lambda t: f(x - t) * t * x, [0, x - 1]); R = mp.quad(lambda t: f(t) * (x ** 2 - t * x), [1, x])
            ok1 &= abs(L - R) < 1e-20
    oks.append(ok1); msgs.append(f"1-1 [증명-수치확인]={ok1}")
    # 1-2: F(x)=sin(ax^2)+cos(bx^2)+√2 는 F(1)=0, F'(1)=0 (좌변 ∫_1^x … 의 값과 도함수가 x=1 에서 0) — b=-3π/2-a 로 a 를 넓게 탐색
    G = lambda a: [mp.sin(a) + mp.cos(-3 * mp.pi / 2 - a) + mp.sqrt(2), 2 * a * mp.cos(a) - 2 * (-3 * mp.pi / 2 - a) * mp.sin(-3 * mp.pi / 2 - a)]
    sols = set()
    for i in range(-400, 401):
        a0 = mp.mpf(i) / 10
        try:
            r = mp.findroot(lambda a: G(a)[0], a0)   # 첫 식의 근 후보 → 둘째 식 확인
        except Exception:
            continue
        if abs(G(r)[0]) < 1e-25 and abs(G(r)[1]) < 1e-20:
            sols.add(mp.nstr(r, 12))
    a_, b_ = load_answer(1, 1)
    oks.append(sols == {mp.nstr(sp.N(a_, 30), 12)} and sp.simplify(a_ + b_ + sp.Rational(3, 2) * sp.pi) == 0)
    msgs.append(f"1-2 a 해={sols} (b=-3π/2-a)")
    report(1, all(oks), "; ".join(msgs))


def p02():
    oks, msgs = [], []
    x = sp.Symbol("x", positive=True)
    f = x * sp.log(x) + x / 2
    ok_all = True
    for r in [sp.Rational(3, 2), sp.Integer(2), sp.Rational(7, 3)]:
        # x_n: 접선 y절편 = -r^{n-1} 을 만족하는 x>0 을 수치로 직접 풂
        xs = {}
        for n in [1, 2, 19, 20, 21]:
            target = -r ** (n - 1)
            F = sp.lambdify(x, f - x * sp.diff(f, x) - target, "mpmath")
            xs[n] = mp.findroot(F, mp.mpf(float(r)) ** (n - 1) * mp.mpf("1.3"))
        v1 = mp.log(xs[21]) / mp.log(xs[20])
        A = lambda n: mp.quad(lambda t: abs(t * mp.log(t) + t / 2), [mp.mpf(float(r)) ** (n - 1), mp.mpf(float(r)) ** n])
        rr = mp.mpf(sp.N(r, 30))
        v2 = (A(20) / (rr ** 38 * mp.log(rr)) - mp.mpf(1) / 2) / (rr ** 2 - 1)
        An = [A(n) for n in range(1, 21)]
        B = lambda m: mp.fsum(An[m - 1:]) / mp.log(rr)
        v3 = B(1) / (2 * rr ** 40)
        v4 = mp.fsum((B(1) - B(m)) / rr ** (2 * m - 2) for m in range(1, 21))
        a1, = load_answer(2, 0); a2, = load_answer(2, 1); a3, a4 = load_answer(2, 2)
        ok = abs(v1 - sp.N(a1, 30)) < 1e-15 and abs(v2 - a2) < 1e-10 and abs(v3 - a3) < 1e-10 and abs(v4 - a4) < 1e-6
        ok_all &= ok
        msgs.append(f"r={r}: {mp.nstr(v1,10)},{mp.nstr(v2,10)},{mp.nstr(v3,10)},{mp.nstr(v4,10)}")
    oks.append(ok_all)
    report(2, all(oks), "; ".join(msgs))


def p03():
    oks, msgs = [], []
    x = sp.Symbol("x")
    pairs = [(p, q) for p in range(1, 7) for q in range(1, 7)]
    nroots = lambda p, q: len(set(r for r in sp.solve(x ** 2 - 2 * p * x + q, x) if r.is_real))
    EX = sp.Rational(sum(nroots(p, q) for p, q in pairs), 36)
    (a1,) = load_answer(3, 0); oks.append(EX == a1); msgs.append(f"3-1 E(X)={EX}")
    A = [(p, q) for p, q in pairs if nroots(p, q) == 2]
    def diff(p, q):
        r = sorted(sp.solve(x ** 2 - 2 * p * x + q, x), key=lambda t: sp.N(t))
        return r[1] - r[0]
    B = [(p, q) for p, q in A if sp.N(diff(p, q)) > 4]
    pr = sp.Rational(len(B), len(A)); (a2,) = load_answer(3, 1)
    oks.append(pr == a2); msgs.append(f"3-2 {len(B)}/{len(A)}")
    # 3-3: n 별로 직접 열거 후 n·P(E_n) 의 정확한 식을 맞춰 극한
    vals = {}
    for n in range(2, 30):
        N = n * n
        cnt = sum(1 for p in range(1, N + 1) for q in range(1, N + 1) if q - p * p > n)
        vals[n] = sp.Rational(n * cnt, N * N)
    nn = sp.Symbol("n")
    poly = sp.interpolate([(n, vals[n] * n ** 3) for n in range(10, 18)], nn)   # n^3·nP(E_n) 은 n 의 다항식
    ok_fit = all(poly.subs(nn, n) == vals[n] * n ** 3 for n in range(2, 30))
    L = sp.limit(poly / nn ** 3, nn, sp.oo); (a3,) = load_answer(3, 2)
    oks.append(ok_fit and L == a3); msgs.append(f"3-3 열거 적합={ok_fit} 극한={L}")
    report(3, all(oks), "; ".join(msgs))


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
