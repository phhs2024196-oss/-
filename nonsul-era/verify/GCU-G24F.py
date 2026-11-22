#!/usr/bin/env python3
"""가천대 2024학년도 논술전형 자연F 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = "GCU-G24F"
WIDTH = 2
RESULTS = []
x, t, a, b, k, n, r = sp.symbols("x t a b k n r", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int) -> str:
    """YAML 의 answer.value (문자열)."""
    it = yaml.safe_load((ROOT / "items/GCU/PAST" / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def num(s: str):
    return sp.nsimplify(sp.sympify(s))


def nums(s: str):
    """답 문자열에서 동그라미 번호를 뺀 수들을 차례로."""
    s = re.sub(r"[①②③④]", " ", s)
    return [sp.Rational(v) for v in re.findall(r"-?\d+(?:/\d+)?", s)]


def p07():
    A = sp.Symbol("A", positive=True)
    y = sp.log(x - 1) / sp.log(A)
    dec = sp.diff(y, x).subs(A, sp.Rational(1, 2)).subs(x, 3) < 0           # ① 0<a<1 에서 감소
    pt = sp.solve(sp.Eq(sp.log(x - 1), 0), x)                                  # ② y=0 인 점
    refl = -y                                                                  # ③ x축 대칭
    alt = sp.log(x - 1) / sp.log(1 / A)
    shift = y.subs(x, x - 3) + 2                                               # ④ 평행이동
    s = ans(7)
    ok = (dec and pt == [2] and sp.simplify(refl - alt) == 0
          and sp.simplify(shift - (sp.log(x - 4) / sp.log(A) + 2)) == 0
          and "감소" in s and "2,0" in s and "\\log_a(x-4)+2" in s and "-\\log_a(x-1)" in s)
    report(7, ok, f"① 감소, ② (2,0), ③ -log_a(x-1)=log_(1/a)(x-1), ④ {sp.simplify(shift)}")


def p08():
    A, B = sp.symbols("A B")
    f = 3 * x**2 - 4 * x + 2 * x * B + A
    Aval = sp.limit(f.subs(x, 1 / t), t, sp.oo)            # (가): lim f(1/x) = f(0)
    Av = sp.solve(sp.Eq(Aval, -3), A)[0]
    Bv = sp.solve(sp.Eq(B, sp.integrate(f.subs(A, Av).subs(x, t), (t, 0, 2))), B)[0]
    F = f.subs({A: Av, B: Bv})
    val = sp.diff(F, x).subs(x, Av)
    report(8, val == num(ans(8)), f"a={Av}, ∫f={Bv}, f={sp.expand(F)}, f'(a)={val}")


def p09():
    A = sp.Symbol("A", positive=True)
    Av = sp.solve(sp.Eq(A**3 - A**2 - 6 * A, 0), A)        # x=a 대입
    f = sp.diff(x**3 - x**2 - 6 * x, x)
    val = f.subs(x, Av[0])
    report(9, len(Av) == 1 and val == num(ans(9)), f"a={Av}, f={f}, f(a)={val}")


def p10():
    X = sp.Symbol("X", positive=True)
    f = lambda z: z**sp.Rational(1, 2) / z**sp.Rational(2, 7)
    e = sp.simplify(sp.log(f(f(X))**49, X))                  # 지수
    best = None
    for m in range(1, 20000):
        v = sp.Integer(m) ** e
        if v.is_integer and v > 1000:
            best = m
            break
    report(10, best == num(ans(10)), f"(f∘f)^49 = n^{e}, 최소 n={best}")


def p11():
    A = sp.Symbol("A", positive=True)
    S = lambda m: sp.Rational(3, 2) * A * m**2 - sp.Rational(1, 2) * A * m
    an = sp.expand(S(n) - S(n - 1))
    term = lambda m: an.subs(n, m)
    tot = sum(7 / (term(2 * j) * term(2 * j + 2)) for j in range(1, 5))
    Av = sp.solve(sp.Eq(tot, 1), A)
    val = term(5).subs(A, Av[0])
    report(11, len(Av) == 1 and val == num(ans(11)), f"a_n={an}, a1={Av}, a5={val}")


def p12():
    c3, c2, c1 = sp.symbols("c3 c2 c1")
    f = x**4 + c3 * x**3 + c2 * x**2 + c1 * x              # (나) x=0: f(0)=0
    sol = sp.solve([f.subs(x, 1), sp.diff(f, x).subs(x, 1) - 4], [c3, c2], dict=True)[0]
    F2 = sp.expand(f.subs(sol).subs(x, 2))                 # f(2) (c1 에 대한 일차식)
    # (나) ⇔ |f'(0)| = |c1| ≤ 4  (x→0 조임; 역으로 g=-(|f/x|+4-|c1|) 가 존재)
    vals = [F2.subs(c1, v) for v in (-4, 4)]
    hi, lo = max(vals), min(vals)
    got = nums(ans(12))
    report(12, got == [hi, lo], f"f(2)={F2}, f'(0)=c1∈[-4,4] → 최댓값 {hi}, 최솟값 {lo}")


def p13():
    K = sp.Symbol("K")
    ok_deg = []
    for deg in range(1, 6):
        cs = sp.symbols(f"q0:{deg + 1}")
        f = sum(cs[i] * x**i for i in range(deg + 1))
        ident = sp.expand(x * sp.diff(f, x) - sp.diff((sp.Rational(2, 3) * x + 1) * (f + K), x))
        sols = sp.solve(sp.Poly(ident, x).coeffs(), list(cs) + [K], dict=True)
        if any(s.get(cs[deg], cs[deg]) != 0 for s in sols):
            ok_deg.append(deg)
    A = sp.Symbol("A", positive=True)
    f = A * (x - 3)**2 - 9                                  # n=2, b=-6a, k=9a-c, 최솟값 -k=-9
    kval = 9
    lhs = sp.integrate(t * sp.diff(f, x).subs(x, t), (t, -sp.Rational(3, 2), x))
    chk = sp.simplify(lhs - (sp.Rational(2, 3) * x + 1) * (f + kval)) == 0
    amin = sp.solve(sp.Eq(f.subs(x, 0), 0), A)[0]          # f(0)≥0 ⇔ a≥1
    I = sp.integrate(f, (x, 0, 6))
    ok_inc = sp.diff(I, A) > 0
    got = nums(ans(13))
    report(13, ok_deg == [2] and chk and ok_inc and got == [2, kval, amin, I.subs(A, amin)],
           f"가능 차수={ok_deg}, (가) 항등 확인={chk}, ∫=18a-54 → a={amin}, 최솟값 {I.subs(A, amin)}")


def p14():
    m = sp.Symbol("m", negative=True)
    mv = sp.solve(sp.Eq(13 / sp.sqrt(1 + m**2), 5), m)[0]  # 직선 y=mx+13, 원점까지 거리 5
    th = sp.pi + sp.atan(mv)                                 # π/2<θ<π
    val = sp.simplify(2 * sp.sin(th) - 3 * sp.cos(th))
    report(14, val == num(ans(14)), f"기울기={mv}, sinθ={sp.simplify(sp.sin(th))}, cosθ={sp.simplify(sp.cos(th))}, 값={val}")


def p15():
    F = sp.Symbol("F")
    P = F**4 - 4 * x**2 * F**2 - 4 * F**2 + 16 * x**2
    fac = sp.factor(P)
    f = sp.Piecewise((-2 * sp.Abs(x), sp.Abs(x) <= 1), (-2, True))
    grid = [sp.Rational(i, 50) for i in range(-200, 201)]
    sat = all(sp.simplify(P.subs(F, f.subs(x, g)).subs(x, g)) == 0 for g in grid)
    mx = max(f.subs(x, g) for g in grid)
    mn = min(f.subs(x, g) for g in grid)
    L = sp.limit(2 * x / x, x, 0, "-")
    R = sp.limit(-2 * x / x, x, 0, "+")
    got = nums(ans(15))
    report(15, sat and mx == 0 and mn == -2 and got == [L, R],
           f"인수분해 {fac}; f=-2|x|(|x|≤1), -2 → 좌극한 {L}, 우극한 {R} [식 성립·최대최소는 격자 확인]")


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
