#!/usr/bin/env python3
"""KNU-G25B (경북대 2025 수시 논술 자연계열Ⅱ 수학 1·2) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "KNU-G25B"
WIDTH = 2
RESULTS = []
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return [sp.sympify(s.strip()) for s in str(v).split(";")]


# ── 문제 1: 조건 (Ⅰ)~(Ⅵ)을 만족하는 f 를 후보 전수 + 수치 판정으로 찾는다 ──
def make_f(a, kap, s1, m1, s2, m2):
    """[5a,7a]: f7+s1(x-7a)^2/m1, [7a,9a]: f7+s2(x-7a)^2/m2 (Ⅲ의 두 경우 x 부호 조합), x>=9a 직선."""
    k = kap / a ** 2
    f5 = -k * 5 * a * (2 * a) ** 2 + 10 * a
    f7 = f5 - s1 * (2 * a) ** 2 / m1
    f9 = f7 + s2 * (2 * a) ** 2 / m2
    c = 2 * s2 * (2 * a) / m2
    def f(x):
        if x <= 5 * a: return -k * x * (x - 3 * a) ** 2 + 10 * a
        if x <= 7 * a: return f7 + s1 * (x - 7 * a) ** 2 / m1
        if x <= 9 * a: return f7 + s2 * (x - 7 * a) ** 2 / m2
        return f9 + c * (x - 9 * a)
    return f, dict(k=k, f5=f5, f7=f7, f9=f9, c=c)


def conditions(a, kap, combo, H=1e-7):
    f, P = make_f(a, kap, *combo)
    if not (P["c"] < 0): return False                                   # (Ⅴ)
    if not (3 * a < P["f5"] < 7 * a): return False                      # (Ⅱ) 값 조건
    dl, dr = (f(5 * a) - f(5 * a - H)) / H, (f(5 * a + H) - f(5 * a)) / H
    if abs(dl - dr) < 1e-4: return False                                # (Ⅱ) 5a 에서 미분불가능
    for x0 in (7 * a, 9 * a):                                           # (Ⅱ) 다른 점은 미분가능
        if abs((f(x0) - f(x0 - H)) - (f(x0 + H) - f(x0))) / H > 1e-4: return False
    if not abs(P["f9"] - P["f5"]) > 0.3 * a ** 2: return False          # (Ⅳ)
    # (Ⅵ) f(f(x)) 미분가능: 문제될 수 있는 점 = x=5a 와 f(x)=5a 인 x 들
    F = lambda x: f(f(x))
    k = P["k"]
    roots = [r.real for r in np.roots([-k, 6 * a * k, -9 * a * a * k, 10 * a - 5 * a]) if abs(r.imag) < 1e-9 and r.real <= 5 * a + 1e-12]
    xs = np.linspace(5 * a, 9 * a, 4001)
    vals = [f(x) - 5 * a for x in xs]
    for i in range(len(xs) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0: roots.append(xs[i])
    if P["c"] != 0: roots.append(9 * a + (5 * a - P["f9"]) / P["c"])
    for x0 in set([5 * a] + [r for r in roots if r >= 9 * a or r <= 9 * a]):
        L, R = (F(x0) - F(x0 - H)) / H, (F(x0 + H) - F(x0)) / H
        if abs(L - R) > 1e-3: return False
    return True


def p01():
    oks, msgs = [], []
    combos = list(itertools.product([1, -1], [25, 81], [1, -1], [25, 81]))
    kaps = [i / 1000 for i in range(1, 1001)]           # ka^2 격자(0.001~1)
    sols = {}
    for a in (1.0, 2.0, 3.0):
        sols[a] = [(cb, kp) for cb in combos for kp in kaps if conditions(a, kp, cb)]
    uniq = all(len(v) == 1 for v in sols.values()) and len({v[0] for v in sols.values()}) == 1
    oks.append(uniq)
    combo, kap = sols[1.0][0]
    msgs.append(f"조건 만족 (부호·분모, ka^2) = {sols[1.0]} (a=1,2,3 동일={uniq})")
    # 1-1
    r1, r2 = load_answer(1, 0)
    a = 2.0; f, P = make_f(a, kap, *combo)
    oks.append(abs((f(5 * a) - f(10 * a)) / a ** 2 - float(r1)) < 1e-12 and abs(kap - float(r2)) < 1e-12)
    msgs.append(f"(f5a-f10a)/a^2={(f(5*a)-f(10*a))/a**2:.6f}, ka^2={kap}")
    # 1-2 p(x)=-lim (f(x-|h|)-f(x))/|h|  → 수치(좌측 차분)
    H = 1e-7
    xs = np.linspace(0, 5 * a, 5001)
    pv = [-(f(x - H) - f(x)) / H for x in xs]
    mx, mn = load_answer(1, 1)
    oks.append(abs(max(pv) - float(mx)) < 1e-4 and abs(min(pv) - float(mn)) < 1e-4)
    msgs.append(f"p 최대={max(pv):.5f} 최소={min(pv):.5f}")
    # 1-3(1) f(17a)=0 → a, c 를 기호로
    A = sp.symbols("A", positive=True)
    s1, m1, s2, m2 = combo
    k = sp.nsimplify(kap) / A ** 2
    f5 = -k * 5 * A * (2 * A) ** 2 + 10 * A
    f7 = f5 - s1 * (2 * A) ** 2 / m1
    f9 = f7 + s2 * (2 * A) ** 2 / m2
    c = sp.Rational(2 * s2 * 2, m2) * A
    asol = sp.solve(sp.Eq(f9 + c * 8 * A, 0), A)
    (cans,) = load_answer(1, 2)
    oks.append(len(asol) == 1 and sp.simplify(c.subs(A, asol[0]) - cans) == 0)
    msgs.append(f"a={asol}, c={[c.subs(A, s) for s in asol]}")
    # 1-3(2) a=25/8 에서 (f∘f)' 최솟값 → 자연수 n 최솟값
    a = float(asol[0]); f, P = make_f(a, kap, *combo)
    F = lambda x: f(f(x))
    h = 1e-6
    xs = np.linspace(-3 * a, 25 * a, 280001)
    d = np.array([(F(x + h) - F(x - h)) / (2 * h) for x in xs])
    dmin = d.min()
    n = 1
    while not (10 * dmin + n > 0): n += 1
    (nans,) = load_answer(1, 3)
    oks.append(n == int(nans))
    msgs.append(f"min (f∘f)'={dmin:.5f} → n_min={n}")
    report(1, all(oks), "; ".join(msgs))


def p02():
    oks, msgs = [], []
    n_, h_ = sp.symbols("n h", positive=True)
    # 2-1(1): 밑면 점 X(P 로부터 d, 경계까지 a)에서 D 의 높이 = (a-d)^2/(2a^2) (Ⅲ 의 곡선, x=a-d).
    # 평면 높이 h 로 자른 아래쪽 부피 = ∬ min(높이, h) dA ; 극좌표(P 중심) u=d/a 로 a^2 가 분리 → 밑면넓이·2∫0^1 min((1-u)^2/2,h) u du
    def V_num(n, h):
        base = mp.mpf(n) / 2 * mp.sin(2 * mp.pi / n)
        u0 = 1 - mp.sqrt(2 * h)                        # (1-u)^2/2 = h 인 u
        I = mp.quad(lambda u: h * u, [0, u0]) + mp.quad(lambda u: (1 - u) ** 2 / 2 * u, [u0, 1])
        return base * 2 * I
    (Vexpr,) = load_answer(2, 0)
    ok = all(abs(V_num(n, mp.mpf(sp.Rational(h).p) / sp.Rational(h).q) - sp.N(Vexpr.subs({sp.Symbol("n"): n, sp.Symbol("h"): sp.Rational(h)}), 30)) < 1e-20
             for n in (3, 4, 7, 12) for h in ("1/20", "1/4", "2/5"))
    oks.append(ok); msgs.append(f"V(n,h) 12점 일치={ok}")
    (lim,) = load_answer(2, 1)
    Lnum = mp.limit(lambda N: V_num(N, mp.mpf(1) / 4), mp.inf)
    oks.append(abs(Lnum - sp.N(lim, 30)) < 1e-12); msgs.append(f"lim V={mp.nstr(Lnum,12)} md={sp.N(lim,12)}")
    # 2-2: S(k) 를 정의대로(곡선 길이 f, 역함수 f^{-1}) 수치 적분
    def S_num(k, n):
        k = mp.mpf(k)
        f = lambda t: mp.quad(lambda x: mp.sqrt(1 + x ** 2 / k ** 4), [0, t])
        finv = lambda s: mp.findroot(lambda t: f(t) - s, s)
        fk = f(k)
        I = mp.quad(lambda s: k - finv(s), [0, fk])
        return 2 / k * mp.sin(mp.pi / n) * I
    mp.mp.dps = 20
    s3 = S_num(mp.sqrt(3), 6)
    (a3,) = load_answer(2, 2)
    oks.append(abs(s3 - sp.N(a3, 20)) < 1e-12); msgs.append(f"S(√3)={mp.nstr(s3,12)} md={sp.N(a3,12)}")
    # (2) k→0+ : 작은 k 들에서 sin(π/n)/S(k) (n=5) 의 수렴값
    vals = [mp.sin(mp.pi / 5) / S_num(kk, 5) for kk in ("1e-3", "1e-4", "1e-5")]
    (a4,) = load_answer(2, 3)
    oks.append(abs(vals[-1] - float(a4)) < 1e-6); msgs.append(f"sin(π/n)/S(k) k=1e-3..1e-5: {[mp.nstr(v,8) for v in vals]}")
    # (3) P=중심: a_i=cos(π/n), T_n=n S(cos π/n) ; n 큰 값으로 수렴 확인
    Ts = [n * S_num(mp.cos(mp.pi / n), n) for n in (200, 2000)]
    (a5,) = load_answer(2, 4)
    oks.append(abs(Ts[-1] - sp.N(a5, 20)) < 1e-4); msgs.append(f"T_200,T_2000={[mp.nstr(v,10) for v in Ts]} md={sp.N(a5,10)}")
    mp.mp.dps = 30
    report(2, all(oks), "; ".join(msgs))


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
