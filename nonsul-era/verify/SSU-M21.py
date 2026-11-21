#!/usr/bin/env python3
"""숭실대 2021학년도 모의논술 (SSU-M21-01·02·04; 3번 확통 제외) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수(번호 = 원문 문제 번호)를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M21"
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
    """g 를 f(y)=x 의 수치해로 정의하고 (1) h 의 수치 미분으로 접선, (2) h 의 수치적분으로 넓이."""
    import mpmath as mp
    mp.mp.dps = 30
    f = lambda y: mp.sqrt(y) * mp.log(y)
    g = lambda X: mp.findroot(lambda y: f(y) - X, mp.mpf(1) + X) if X > 0 else mp.mpf(1)
    h = lambda X: g(2 * X - 1)
    x0 = (mp.sqrt(mp.e) + 1) / 2
    y0 = h(x0)
    slope = mp.diff(h, x0)
    ans1 = load_answer(1, 0, {"x": x})
    line = lambda X: mp.mpf(str(sp.N(ans1.subs(x, sp.Float(str(X), 40)), 40)))
    ok1 = abs(y0 - mp.e) < 1e-20 and all(abs(line(X) - (y0 + slope * (X - x0))) < 1e-15 for X in (x0, x0 + 1, mp.mpf(-2)))
    S_num = mp.quad(h, [mp.mpf(1) / 2, x0])
    ans2 = load_answer(1, 1)
    ok2 = abs(S_num - mp.mpf(str(sp.N(ans2, 40)))) < 1e-15
    report(1, ok1 and ok2, f"(1) h(x0)={mp.nstr(y0, 12)}, 기울기={mp.nstr(slope, 15)} (2) S={mp.nstr(S_num, 15)}, md={ans1}, {ans2}")


def p02():
    """반지름²=u²+(u tan u)², 중심각 π/2-u, V≡1 → h=1/A. dh/du 의 u→π/2- 극한(기호 + 고정밀 수치)."""
    import mpmath as mp
    u = sp.symbols("u", positive=True)
    r2 = u ** 2 + (u * sp.tan(u)) ** 2
    A = r2 * (sp.pi / 2 - u) / 2
    hfun = 1 / A
    dh = sp.diff(hfun, u)
    s_ = sp.symbols("s", positive=True)
    lim = sp.limit(sp.simplify(dh.subs(u, sp.pi / 2 - s_)), s_, 0, "+")
    mp.mp.dps = 50
    fnum = sp.lambdify(u, dh, "mpmath")
    num = fnum(mp.pi / 2 - mp.mpf("1e-20"))
    ans = load_answer(2)
    ok = sp.simplify(lim - ans) == 0 and abs(num - mp.mpf(str(sp.N(ans, 50)))) < 1e-15
    report(2, ok, f"기호 극한={lim}, 수치={mp.nstr(num, 15)}, md={ans}")


def p04():
    """θ_n = y축과 이루는 각 → θ_n = atan(n/(1/n^k)). k=1..5 에서 md 의 (a,b) 로 큰 n 의 값이 1 에 수렴, b±1 이면 0/발산."""
    import mpmath as mp
    mp.mp.dps = 80
    K = sp.Symbol("k")
    ans = load_answer(4, 0, {"k": K})
    msgs, ok = [], True
    for k in range(1, 6):
        a_, b_ = float(ans[0].subs(K, k)), float(ans[1].subs(K, k))
        th = lambda N: mp.atan(mp.mpf(N) / (mp.mpf(1) / mp.mpf(N) ** k))
        val = lambda N, bb: a_ * mp.mpf(N) ** bb * mp.tan(th(N + 1) - th(N))
        v = val(10 ** 9, b_)
        ok &= abs(v - 1) < 1e-7 and abs(val(10 ** 9, b_ - 1)) < 1e-7 and abs(val(10 ** 9, b_ + 1)) > 1e7
        msgs.append(f"k={k}:{mp.nstr(v, 10)}")
    report(4, ok, f"{', '.join(msgs)}, md={ans}")


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
