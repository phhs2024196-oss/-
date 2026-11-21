#!/usr/bin/env python3
"""숭실대 2024학년도 논술고사(자연) 1~4 (SSU-G24-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G24"
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
        f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """두 원(중심·반지름)에 대한 접선 조건 = 거리 조건을 기호로 풀어 기울기 음수인 접선을 구하고, y절편을 적분."""
    tt = sp.symbols("tt", positive=True)
    m, c = sp.symbols("m c", real=True)
    c1, r1 = (tt - 1) / 2, (1 + tt) / 2
    c2, r2 = (tt + 1) / 2, (1 - tt) / 2
    sols = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            for sol in sp.solve([sp.Eq(s1 * (m * c1 + c), r1 * sp.sqrt(1 + m**2)),
                                 sp.Eq(s2 * (m * c2 + c), r2 * sp.sqrt(1 + m**2))], [m, c], dict=True):
                sols.append(sol)
    ok1, checked = True, 0
    ans1 = load_answer(1, 0).subs(t, tt)
    for tv in (sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(3, 5), sp.Rational(9, 10)):
        neg = {(sp.simplify(s[m].subs(tt, tv)), sp.simplify(s[c].subs(tt, tv))) for s in sols
               if s[m].subs(tt, tv).is_real and s[m].subs(tt, tv) < 0}
        expect = (sp.simplify(sp.diff(ans1, x).subs(tt, tv)), sp.simplify(ans1.subs({x: 0, tt: tv})))
        ok1 &= len(neg) == 1 and all(sp.simplify(p - q) == 0 for p, q in zip(neg.pop(), expect))
        checked += 1
    # y절편 f(t): 기울기 음수인 해의 c 를 기호로
    cexpr = [sp.simplify(s[c]) for s in sols if sp.simplify(s[m].subs(tt, sp.Rational(1, 2))) < 0][0]
    import mpmath as mp
    mp.mp.dps = 40
    Inum = mp.quad(sp.lambdify(tt, cexpr, "mpmath"), [mp.mpf(1) / 2, mp.sqrt(2) / 2])
    I = sp.nsimplify(sp.N(Inum, 35), [sp.pi, sp.sqrt(3)])      # 고정밀 수치 → π, √3 기저로 식 복원(참고 표시용)
    ans2 = load_answer(1, 1)
    ok2 = abs(Inum - sp.N(ans2, 40)) < mp.mpf(10) ** -30
    report(1, ok1 and ok2, f"(1) t {checked}개 값에서 접선 일치={ok1}, f(t)={cexpr}; (2) 적분≈{mp.nstr(Inum,20)} (복원 {I}), md={ans2}")


def p02():
    """(1) [증명-수치확인] 여러 k, n 에서 u 의 증감을 격자로 확인. (2) k 격자에서 u 의 영점을 직접 세어 md 범위와 비교."""
    import numpy as np
    def hfun(X, kv):
        return (-kv + np.sqrt(kv * kv - 4 + 4 * X)) / 2          # g(y)=X 의 y≥0 근
    def ffun(X):
        nn = np.floor(X / 2)
        return 1 - np.abs(X - 2 * nn - 1)
    mono_ok = True
    for kv in (1.5, 3.0, 50.0, 1000.0):
        for nn in (1, 2, 7, 300):
            xs = np.linspace(2 * nn, 2 * nn + 1, 2001)
            xs2 = np.linspace(2 * nn + 1, 2 * nn + 2, 2001)[:-1]
            uL = hfun(xs, kv) - ffun(xs)
            uR = hfun(xs2, kv) - ffun(xs2)
            mono_ok &= bool(np.all(np.diff(uL[:-1]) < 0) and np.all(np.diff(uR) > 0))
    def count_zeros(kv):
        cnt = 0
        X = np.linspace(1, 2, 20001)
        U = hfun(X, kv) - ffun(np.minimum(X, 1.999999999))
        cnt += int(np.sum(np.sign(U[:-1]) * np.sign(U[1:]) < 0))
        from fractions import Fraction
        K = Fraction(kv).limit_denominator(1000)
        for nn in range(1, 700):                                  # [2n, 2n+2]: 감소-증가, 극솟값 u(2n+1)=h(2n+1)-1 의 부호
            # h(X)-1 = (sqrt(K^2-4+4X) - (K+2))/2, 양변 양수이므로 제곱 비교(정확한 유리수 연산)
            d = (K * K - 4 + 4 * (2 * nn + 1)) - (K + 2) ** 2
            cnt += 2 if d < 0 else (1 if d == 0 else 0)
        return cnt
    ks = [sp.Rational(j, 2) for j in range(1994, 2006)]
    ans = load_answer(2, 1)
    res = {kv: count_zeros(float(kv)) for kv in ks}
    bad = [kv for kv in ks if (res[kv] == 1001) != bool(ans.contains(kv))]
    report(2, mono_ok and not bad, f"(1) [증명-수치확인] 증감 {mono_ok}; (2) k별 교점 수 {[(str(kv), c) for kv, c in res.items()]}, md={ans}")


def p03():
    """f(g(x)) 연속에서 a 를 풀고, 높이 y 에 대한 직사각형 넓이(폭 = 두 곡선의 x 차)를 최대화."""
    av = sp.symbols("av", real=True)
    f = lambda X: sp.Piecewise((1 - X**2, X <= 0), (1 - av * X**2, True))
    g = lambda X: sp.Piecewise((-X + sp.Rational(1, 2), X <= 0), (-X - sp.Rational(1, 4), True))
    eps = sp.symbols("eps", positive=True)
    left = sp.limit(f(g(-eps)), eps, 0)
    right = sp.limit(f(g(eps)), eps, 0)
    val = f(g(0))
    a0 = sp.solve([sp.Eq(left, val), sp.Eq(right, val)], av)
    a0 = a0[av] if isinstance(a0, dict) else a0[0]
    y = sp.symbols("y", positive=True)
    width = sp.sqrt((1 - y) / a0) + sp.sqrt(1 - y)                  # 오른쪽 x - 왼쪽 x
    S = sp.simplify(width * y)
    crit = [cv for cv in sp.solve(sp.diff(S, y), y) if 0 < cv < 1]
    Smax = max(S.subs(y, cv) for cv in crit)
    ans = load_answer(3)
    ok = a0 == sp.Rational(1, 4) and sp.simplify(Smax - ans) == 0
    report(3, ok, f"a={a0}, 넓이(높이 y)={S}, 최댓값={sp.nsimplify(Smax)}, md={ans}")


def p04():
    """6^6 개 함수 전수 열거로 f∘f=id 개수, (1) 학생 A 의 두 오류를 수치로 확인 [증명-수치확인]."""
    import itertools, math
    dom = range(6)
    invs = [fn for fn in itertools.product(dom, repeat=6) if all(fn[fn[i]] == i for i in dom)]
    fixed = sum(1 for fn in invs if any(fn[i] == i for i in dom))
    nofix = len(invs) - fixed
    student = math.comb(6, 2) * math.comb(4, 2) * math.comb(2, 2)
    ok1 = fixed > 0 and student == nofix * math.factorial(3)     # 오류1: 고정점 있는 함수 존재, 오류2: 3! 중복
    ans = load_answer(4, 1)
    report(4, ok1 and len(invs) == ans, f"(1) [증명-수치확인] 고정점 있는 함수 {fixed}개, 학생 90 = 고정점 없음 {nofix}×3!; (2) 전수 {len(invs)}, md={ans}")


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
