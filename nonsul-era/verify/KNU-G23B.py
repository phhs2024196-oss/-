#!/usr/bin/env python3
"""KNU 2023학년도 수시 논술(AAT) 자연계열Ⅱ 수학 (KNU-G23B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import math
import random
import re
import sys

import sympy as sp

PREFIX = "KNU-G23B"
WIDTH = 2
RESULTS = []
x, y, t, u = sp.symbols("x y t u", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())


def load_answer(seq, i=0):
    return load_item(seq)["subparts"][i]["answer"]["value"]


def p01():
    msgs, ok = [], True
    N = 12
    A = sp.symbols(f"a1:{N + 2}"); B = sp.symbols(f"b1:{N + 2}"); C = sp.symbols(f"c1:{N + 2}")
    piece = lambda k, xx: A[k - 1] * (xx - 2 * k + 1) ** 4 + B[k - 1] * (xx - 2 * k + 1) ** 2 + C[k - 1] * (xx - 2 * k + 1) + sp.Rational(1, 2 ** (k - 1))
    h = sp.symbols("h", positive=True)
    eqs = []
    for k in range(1, N + 1):
        # x=2k-1: 우미분 = 좌미분(좌측은 조건 (Ⅱ) f(x)=f(4k-2-x))
        R = sp.limit((piece(k, 2 * k - 1 + h) - piece(k, 2 * k - 1)) / h, h, 0)
        L = sp.limit((piece(k, 2 * k - 1 + h) - piece(k, 2 * k - 1)) / (-h), h, 0)   # f(2k-1-h)=f(2k-1+h)
        eqs.append(sp.Eq(R, L))
        # x=2k: 연속 + 좌미분 = 우미분 (우측 f(2k+s)=f(2k+2-s) 는 k+1 조각)
        eqs.append(sp.Eq(piece(k, 2 * k), piece(k + 1, 2 * k + 2)))
        Lk = sp.diff(piece(k, x), x).subs(x, 2 * k)
        Rk = -sp.diff(piece(k + 1, x), x).subs(x, 2 * k + 2)
        eqs.append(sp.Eq(Lk, Rk))
    # 1-1: c_n 은 (Ⅰ)(Ⅱ)·미분가능만으로 0
    solc = sp.solve(eqs[0::3], C[:N])
    a11 = sp.sympify(load_answer(1, 0))
    ok &= all(solc[C[k]] == a11 for k in range(N))
    msgs.append(f"1-1 c_1..c_{N}={set(solc.values())} md={a11}")
    # 1-2: 무작위 (a_1,b_1) 에서 관계로 전개 → 곱 ≤ 0 [증명-수치확인]
    rel = sp.solve([e.subs(solc) for e in eqs[2::3]][:1], [B[1]])
    good = True
    for k in range(1, N):
        e = sp.Eq(sp.diff(piece(k, x), x).subs(x, 2 * k), -sp.diff(piece(k + 1, x), x).subs(x, 2 * k + 2)).subs(solc)
        bk1 = sp.solve(e, B[k])[0]
        for _ in range(20):
            vals = {A[k - 1]: random.uniform(-5, 5), B[k - 1]: random.uniform(-5, 5), A[k]: random.uniform(-5, 5)}
            good &= float((2 * vals[A[k - 1]] + vals[B[k - 1]]) * (2 * vals[A[k]] + bk1.subs(vals))) <= 1e-12
    ok &= good
    msgs.append(f"1-2 미분가능 조건하 곱≤0 (무작위)={good} [증명-수치확인]")
    # 1-3: b1=-4, f(0+)=f(2) (조건 (Ⅱ), n=1) = -1
    extra = [sp.Eq(B[0], -4), sp.Eq(piece(1, 2), -1)]
    sol = sp.solve(eqs + extra, list(A[:N + 1]) + list(B[:N + 1]) + list(C[:N]), dict=True)[0]
    pieces = {k: sp.lambdify(x, piece(k, x).subs(sol)) for k in range(1, N + 1)}
    def f_num(xx):
        k = math.ceil(xx / 2)                     # (2k-2, 2k]
        if xx < 2 * k - 1: xx = 4 * k - 2 - xx      # 조건 (Ⅱ)
        return pieces[k](xx)
    fp = lambda k: sp.diff(piece(k, x), x).subs(sol)
    I = 0
    for k in range(1, 6):          # [2k-1,2k] 조각과 대칭 조각의 |f'| 적분(정확)
        seg = sp.integrate(sp.Abs(sp.factor(fp(k))), (x, 2 * k - 1, 2 * k))
        if 2 * k - 1 >= 3: I += seg           # [3,4],[5,6],..,[9,10]
        if 2 * k - 1 > 2: I += seg            # 대칭 조각 [2k-2,2k-1]: |f'| 적분 동일
    I = sp.nsimplify(I)
    pq = sp.fraction(I)
    a131 = sp.sympify(load_answer(1, 2))
    ok &= pq[0] + pq[1] == a131
    msgs.append(f"1-3(1) ∫|f'|={I} p+q={pq[0] + pq[1]} md={a131}")
    # 1-3(2): m(t) = f(x)=t 의 횡단 교점 수 (격자 부호변화) ; t=2^-k(1±ε) 에서 m/log2 t 경향
    xs = [i / 2000 for i in range(1, 2000 * 2 * N)]
    fv = [f_num(z) for z in xs]
    def m(tt):
        return sum(1 for i in range(len(fv) - 1) if (fv[i] - tt) * (fv[i + 1] - tt) < 0)
    seq = [(k, m(2.0 ** -k * 1.3), m(2.0 ** -k * 1.3) / math.log2(2.0 ** -k * 1.3)) for k in (3, 6, 9)]
    a132 = sp.sympify(load_answer(1, 3))
    good2 = all(mm == 2 * k for k, mm, _ in seq)
    ok &= good2 and abs(seq[-1][2] - float(a132)) < 0.1 and abs(seq[-1][2] - a132) < abs(seq[0][2] - a132)
    msgs.append(f"1-3(2) m(1.3·2^-k)={[(k, mm) for k, mm, _ in seq]} 비={[round(r, 3) for *_, r in seq]} → md={a132}")
    report(1, ok, "; ".join(msgs))


def p02():
    msgs, ok = [], True
    a = sp.Integer(2023)
    F = sp.Function("F"); C1, c = sp.symbols("C1 c")
    # 2-1/2-2: (Ⅰ)을 y 로 미분해 y=0 → F'(x) = F(x) ln a + a^x F'(0) (=:k). F(0)=0 은 x=y=0 대입에서.
    k = sp.symbols("k")
    sol = sp.dsolve(sp.Eq(F(x).diff(x), F(x) * sp.log(a) + a ** x * k), F(x), ics={F(0): 0}).rhs
    # 해가 실제로 (Ⅰ)을 만족하고 F'(0)=k 인지
    chk1 = sp.simplify(sol.subs(x, x + y) - (a ** y * sol + a ** x * sol.subs(x, y))) == 0
    chk2 = sp.simplify(sol.diff(x).subs(x, 0) - k) == 0
    ok &= chk1 and chk2
    msgs.append(f"2-1 (Ⅰ)⇒ODE 해 {sol} 가 (Ⅰ)·F'(0)=k 만족={chk1 and chk2} [증명-수치확인]")
    kv = sp.solve(sp.Eq(sol.subs(x, a), a), k)[0]
    fx = sp.simplify(sol.subs(k, kv))
    a22 = sp.sympify(load_answer(2, 1), locals={"x": x})
    ok &= sp.simplify(sp.log(fx / x) - sp.log(a22 / x)) == 0 and all(sp.N(fx.subs(x, v) / a22.subs(x, v) - 1, 30) == 0 for v in (1, 2, sp.Rational(1, 3)))
    msgs.append(f"2-2 f={fx} md={a22}")
    # 2-3: G=log_a g, (Ⅱ) ⇒ G(x+y)=xy(2x²+3xy+2y²)+G(x)+G(y). y 미분 후 y=0 → G'(x)=2x³+G'(0), G(0)=0
    Gf = sp.Function("G")
    Gs = sp.dsolve(sp.Eq(Gf(x).diff(x), 2 * x ** 3 + c), Gf(x), ics={Gf(0): 0}).rhs
    chk3 = sp.expand(Gs.subs(x, x + y) - (x * y * (2 * x ** 2 + 3 * x * y + 2 * y ** 2) + Gs + Gs.subs(x, y))) == 0
    cv = sp.solve(sp.Eq(Gs.subs(x, a), 1), c)[0]
    Gx = sp.expand(Gs.subs(c, cv))
    a23 = sp.sympify(load_answer(2, 2), locals={"x": x})
    ex = sp.expand(sp.expand_log(sp.log(a23), force=True) / sp.log(a))
    ok &= chk3 and sp.expand(ex - Gx) == 0
    msgs.append(f"2-3 log_2023 g={Gx} (Ⅱ)만족={chk3} md 지수 일치={sp.expand(ex - Gx) == 0}")
    report(2, ok, "; ".join(msgs))


def p03():
    import mpmath as mp
    mp.mp.dps = 30
    msgs, ok = [], True
    # 3-1: 무작위 감소 h 예시(f=2^(1/x), g=1/(x+3), m=1; f=-ln x, g=(x²+1)/(x+1), m=1)에서 부등식 수치 확인 [증명-수치확인]
    good = True
    for fexpr, gexpr in ((2 ** (1 / x), 1 / (x + 3)), (-sp.log(x), (x ** 2 + 1) / (x + 1))):
        hexpr = fexpr.subs(x, gexpr) * sp.diff(gexpr, x)
        H = sp.lambdify(x, hexpr, "mpmath"); Fn = sp.lambdify(x, fexpr, "mpmath"); G = sp.lambdify(x, gexpr, "mpmath")
        dh = sp.lambdify(x, sp.diff(hexpr, x), "mpmath")
        good &= all(dh(1 + i / 50) < 0 for i in range(1, 500))
        for n in range(1, 12):
            I = mp.quad(Fn, [G(1), G(n + 1)])
            good &= sum(H(i) for i in range(2, n + 2)) < I < sum(H(i) for i in range(1, n + 1))
    ok &= good
    msgs.append(f"3-1 두 예시 h 감소·부등식(n≤11)={good} [증명-수치확인]")
    # 3-2: 직접 수치적분
    v = mp.quad(lambda z: 2 ** (1 / z), [mp.mpf(1) / 7, mp.mpf(1) / 4]) / 2
    a32 = sp.sympify(load_answer(3, 1))
    ok &= int(mp.floor(v)) == a32
    msgs.append(f"3-2 값={mp.nstr(v, 10)} 정수부분={int(mp.floor(v))} md={a32}")
    # 3-3: a_n 을 정의대로 계산(n 큰 값) + 끼움 수열 극한을 sympy 로
    nn = sp.symbols("nn", positive=True)
    Gn = (nn ** 2 + 2 * nn + 2) / (nn + 2)
    Iexpr = -Gn * (sp.log(Gn) - 1) - 1
    bn = 4 / ((nn + 1) * sp.log(nn + 1)) * Iexpr
    cn = (4 / (nn * sp.log(nn)) * Iexpr).subs(nn, nn + 1)
    Lb, Lc = sp.limit(bn, nn, sp.oo), sp.limit(cn, nn, sp.oo)
    import math
    def a(n):
        s = math.fsum((k * k + 2 * k - 1) / (k + 1) ** 2 * math.log((k + 1) / (k * k + 1)) for k in range(1, n + 2))
        return 4 / ((n + 1) * math.log(n + 1)) * s
    sand = all(float(cn.subs(nn, n)) < a(n) < float(bn.subs(nn, n)) for n in (2, 5, 50, 1000, 10 ** 5))
    a33 = sp.sympify(load_answer(3, 2))
    ok &= Lb == a33 and Lc == a33 and sand
    msgs.append(f"3-3 lim b_n={Lb}, lim c_n={Lc}, c_n<a_n<b_n(표본)={sand}, a_1e6={a(10 ** 6):.4f} md={a33}")
    report(3, ok, "; ".join(msgs))


if __name__ == "__main__":
    random.seed(7)
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
