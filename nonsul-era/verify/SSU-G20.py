#!/usr/bin/env python3
"""숭실대 2020학년도 논술 (SSU-G20-01·02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수(번호 = 원문 문제 번호)를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G20"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, locals_=None):
    """yaml 의 subparts[sub].answer.value 를 sympy 로 읽는다."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return sp.sympify(d["subparts"][sub]["answer"]["value"], locals=locals_ or {})


def close(a, b, tol=1e-9):
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(1.0, abs(b))


def p01():
    """f(x)=∫_0^x e^{-t²}dt 를 수치로 두고 (1) A(α), (2) B(α)(역함수 수치적분) 를 md 식과 비교, (3) 극한은 erf 로 기호 계산."""
    import mpmath as mp
    mp.mp.dps = 30
    f = lambda X: mp.quad(lambda s: mp.e ** (-s * s), [0, X])
    al = sp.Symbol("alpha")
    F = sp.Function("f")
    A_md = load_answer(1, 0, {"alpha": al, "f": F})
    B_md = load_answer(1, 1, {"alpha": al})
    errs = []
    for a0 in (mp.mpf("0.3"), mp.mpf(1), mp.mpf("1.7")):
        A_num = mp.quad(f, [0, a0])
        A_val = sp.N(A_md.subs(F(al), sp.Float(str(f(a0)), 30)).subs(al, sp.Float(str(a0), 30)), 30)
        g = lambda y: mp.findroot(lambda X: f(X) - y, y)          # 역함수
        B_num = mp.quad(g, [0, f(a0)])
        B_val = sp.N(B_md.subs(al, sp.Float(str(a0), 30)), 30)
        errs += [abs(A_num - mp.mpf(str(A_val))), abs(B_num - mp.mpf(str(B_val)))]
    a = sp.symbols("a", positive=True)
    fsym = sp.sqrt(sp.pi) / 2 * sp.erf(x)                         # ∫_0^x e^{-t²} dt
    Asym = sp.integrate(fsym, (x, 0, a))
    Bsym = a * fsym.subs(x, a) - Asym                             # 넓이 관계 대신 직접: ∫_0^{f(a)} g = a f(a) - ∫_0^a f
    lim = sp.limit(Asym / Bsym, a, 0, "+")
    ans3 = load_answer(1, 2)
    ok = max(errs) < 1e-15 and sp.simplify(lim - ans3) == 0
    report(1, ok, f"(1)(2) 수치 최대오차={mp.nstr(max(errs), 3)} (3) 극한={lim}, md={A_md}; {B_md}; {ans3}")


def p02():
    """조건을 만족하는 구체적 f=2(1-(1-x)^m) 을 만들고 g 를 (i) 의 적분 + (ii) + 연속성으로 수치 구성.
    (1) g≥0 on [-1,1] 인 α 의 범위를 격자로, (2) 넓이 최대 α 를 황금분할로."""
    import numpy as np
    import mpmath as mp
    mp.mp.dps = 30
    m = mp.findroot(lambda M: 4 * (1 - 2 / (M + 1) + 1 / (2 * M + 1)) - 3, 5)   # ∫f²=3
    f = lambda X: 2 * (1 - (1 - X) ** m)
    fp = lambda X: 2 * m * (1 - X) ** (m - 1)
    assert abs(mp.quad(lambda X: f(X) ** 2, [0, 1]) - 3) < 1e-20 and m > 1
    xs_pos = np.linspace(0, 1, 401)
    I_pos = np.array([float(mp.quad(lambda s: 2 * f(s) * fp(s), [0, X])) for X in xs_pos])   # ∫_0^x 2 f f'
    fneg = np.array([float(f(X)) for X in xs_pos])                                          # f(-x) for x=-X

    def gvals(al):
        g0 = -0.0 + 5 - al ** 3                    # (ii) at x=0, f(0)=0 → g(0); x>0 은 연속성으로 g(0)+α∫2ff'
        gp = g0 + al * I_pos
        gn = -fneg ** 2 + 5 - al ** 3
        return gn, gp

    alphas = np.linspace(0, 2, 2001)
    good = [al for al in alphas if min(gvals(al)[0].min(), gvals(al)[1].min()) >= -1e-12]
    lo, hi = min(good), max(good)
    ans1 = load_answer(2, 0)
    ok1 = abs(lo - float(ans1.inf)) < 1e-9 and abs(hi - float(ans1.sup)) < 1.5e-3 and bool(ans1.contains(sp.Rational(1, 2)))

    def area(al):
        gn, gp = gvals(al)
        return np.trapezoid(np.abs(gn), xs_pos) + np.trapezoid(np.abs(gp), xs_pos)

    l, r = 0.0, float(ans1.sup)
    for _ in range(80):
        c1, c2 = l + (r - l) * 0.382, l + (r - l) * 0.618
        if area(c1) < area(c2):
            l = c1
        else:
            r = c2
    ans2 = load_answer(2, 1)
    ok2 = abs((l + r) / 2 - float(ans2)) < 1e-6
    report(2, ok1 and ok2, f"(1) 격자 범위≈[{lo:.4f},{hi:.4f}] (2) argmax α≈{(l + r) / 2:.8f}, 넓이≈{area((l + r) / 2):.6f}, md={ans1}, {ans2}")


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
