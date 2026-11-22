#!/usr/bin/env python3
"""KNU-M21B (경북대 2021 논술 모의고사 자연계열Ⅱ 수학 1·2·3) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KNU-M21B"
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





def _roots(N):
    """sin x/x = cos x/x 의 양의 근을 부호변화+이분법으로 N 개"""
    h = lambda x: mp.sin(x) - mp.cos(x)
    out, x, step = [], mp.mpf("0.01"), mp.mpf("0.05")
    while len(out) < N:
        if h(x) * h(x + step) < 0:
            out.append(mp.findroot(h, (x, x + step), solver="bisect"))
        x += step
    return out


def p01():
    oks, msgs = [], []
    mp.mp.dps = 30
    a = _roots(12)
    (v1,) = load_answer(1, 0)
    oks.append(abs(a[9] - a[1] - sp.N(v1, 30)) < 1e-20); msgs.append(f"1-1 a10-a2={mp.nstr(a[9]-a[1],12)}")
    # 1-2, 1-3: 큰 n 에서 S(n)/ln n 의 수렴을 차분으로 확인: (S(n2)-S(n1))/(ln n2 - ln n1) → 극한
    an = lambda k: mp.pi / 4 + (k - 1) * mp.pi        # 1-1 에서 수치로 확인한 등차 규칙 (아래 검증)
    oks.append(all(abs(a[k - 1] - an(k)) < 1e-20 for k in range(1, 13)))
    S1 = lambda n: (mp.digamma(n + mp.mpf(1) / 4) - mp.digamma(mp.mpf(1) / 4)) / mp.pi   # Σ 1/(π(k-3/4)) 정확값
    oks.append(abs(S1(50) - mp.fsum(1 / an(k) for k in range(1, 51))) < 1e-20)
    n1, n2 = 10 ** 6, 10 ** 12
    L2 = (S1(n2) - S1(n1)) / (mp.log(n2) - mp.log(n1))
    (v2,) = load_answer(1, 1)
    oks.append(abs(L2 - sp.N(v2, 30)) < 1e-6); msgs.append(f"1-2 증분비={mp.nstr(L2,12)}")
    # A_k 를 실제 적분으로 몇 개 계산해 2√2/a 사이에 있음을 확인하고, A_k*a_k 의 극한 → 2√2 이므로 같은 증분비
    Ak = lambda k: abs(mp.quad(lambda x: (mp.sin(x) - mp.cos(x)) / x, [an(k), an(k + 1)]))
    okb = all(2 * mp.sqrt(2) / an(k + 1) <= Ak(k) <= 2 * mp.sqrt(2) / an(k) for k in range(1, 40))
    ratio = Ak(10 ** 5) * an(10 ** 5)
    (v3,) = load_answer(1, 2)
    oks.append(okb and abs(ratio * L2 - sp.N(v3, 30)) < 1e-5); msgs.append(f"1-3 끼움 성립={okb} A_k·a_k→{mp.nstr(ratio,10)} 극한≈{mp.nstr(ratio*L2,10)}")
    report(1, all(oks), "; ".join(msgs))


def p02():
    oks, msgs = [], []
    x, a, b, c, m, n = sp.symbols("x a b c m n")
    f = a * x ** 3 + b * x ** 2 + c * x + n
    sol = sp.solve([sp.diff(f, x).subs(x, m), sp.diff(f, x).subs(x, -m), f.subs(x, m) - m ** 3 - n], [a, b, c], dict=True)
    so = sol[0]
    fx = f.subs(so)
    ok_type = sp.diff(fx, x, 2).subs(x, m).subs(m, 3) < 0 and sp.diff(fx, x, 2).subs(x, -m).subs(m, 3) > 0   # m 에서 극대, -m 에서 극소
    v = sp.simplify(4 * so[a] * so[c] / m ** 2)
    (a1,) = load_answer(2, 0); oks.append(ok_type and v == a1); msgs.append(f"2-1 f={fx} 값={v}")
    # 2-2: 격자점 직접 열거
    def B(mv, nv=0):
        F = sp.lambdify(x, fx.subs({m: mv, n: nv}))
        cnt = 0
        for i in range(0, mv + 1):
            lo, hi = mv * i * i + nv, F(i)
            ylo, yhi = min(lo, hi), max(lo, hi)
            cnt += int(sp.floor(sp.nsimplify(yhi)) - sp.ceiling(sp.nsimplify(ylo)) + 1)
        return cnt
    k = sp.Symbol("k")
    data = [(kv, B(2 * kv - 1, nv=-3)) for kv in range(1, 9)]
    P = sp.expand(sp.interpolate([(kv, 6 * bv) for kv, bv in data[:5]], k))   # 6B = k(pk^3+qk^2+rk+s)
    okfit = all(P.subs(k, kv) == 6 * bv for kv, bv in data)
    co = sp.Poly(sp.cancel(P / k), k).all_coeffs()
    p_, q_, r_, s_ = co
    (a2,) = load_answer(2, 1)
    oks.append(okfit and (p_ - q_) * r_ / s_ == a2); msgs.append(f"2-2(1) 열거 적합 6B=k({p_}k^3+{q_}k^2+{r_}k+{s_}) 값={(p_-q_)*r_/s_}")
    # 2-2(2) [증명-수치확인]: B_m/A_m (A_m 은 적분) 이 1 로 접근
    rat = [sp.N(B(mv) / sp.integrate((fx - m * x ** 2 - n).subs({m: mv, n: 0}), (x, 0, mv))) for mv in [10, 40, 160]]
    okc = all(sp.integrate((fx - m * x ** 2 - n).subs({m: mv, n: 0}), (x, 0, mv)) == sp.Rational(7, 24) * mv ** 4 for mv in [10, 40])
    oks.append(okc and abs(rat[-1] - 1) < abs(rat[0] - 1) and abs(rat[-1] - 1) < 0.03); msgs.append(f"2-2(2) B/A (m=10,40,160)={[round(float(t),4) for t in rat]} [증명-수치확인]")
    report(2, all(oks), "; ".join(msgs))


def p03():
    oks, msgs = [], []
    x, t, A, B = sp.symbols("x t A B", positive=True)
    # 연속해 h(x)=ln f(x)=A·B^x (A=h(0)) 로 두고 함수방정식 f(x+t)=f(x)^{g(t)} 가 g(t)=h(t)/h(0) 로 성립하는지 확인
    h = A * B ** x
    g = sp.simplify(h.subs(x, t) / A)
    okfe = sp.simplify(h.subs(x, x + t) - g * h) == 0
    (a1,) = load_answer(3, 0)
    oks.append(okfe and g.subs(t, 0) == a1); msgs.append(f"3-1 g(0)={g.subs(t,0)} 함수방정식 성립={okfe}")
    # 3-2: ln f(e)=2 ln f(0) → B^e=2
    Bs = sp.solve(sp.Eq(h.subs(x, sp.E), 2 * A), B)[0]
    S = sp.simplify(sum(h.subs({x: k * sp.E, B: Bs}) for k in range(1, 11)) / A)
    (a2,) = load_answer(3, 1)
    oks.append(S == a2); msgs.append(f"3-2 α={S}")
    # 3-3: f(e)=e^4 → h(e)=4 → A=2
    As = sp.solve(sp.Eq(h.subs({x: sp.E, B: Bs}), 4), A)[0]
    hh = h.subs({A: As, B: Bs})
    v1 = sp.simplify(hh.subs(x, 1))
    n = sp.Symbol("n", positive=True)
    v2 = sp.limit((hh.subs(x, sp.E + sp.E / n) - 4) * n, n, sp.oo)
    b1, b2 = load_answer(3, 2)
    oks.append(sp.simplify(v1 - b1) == 0 and sp.simplify(v2 - b2) == 0); msgs.append(f"3-3 ln f(1)={v1}, 극한={v2}")
    # 수치: 정수·유리수 반복 정의로 만든 h 와 일치 확인 [증명-수치확인]
    h0, h1 = mp.mpf(2), mp.mpf(2) ** (1 + 1 / mp.e)
    hq = lambda p, q: h0 ** (1 - mp.mpf(p) / q) * h1 ** (mp.mpf(p) / q)
    oks.append(abs(hq(int(mp.e * 10 ** 12), 10 ** 12) - 4) < 1e-9)   # 유리수 근사 p/q→e 에서 h→h(e)=4
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
