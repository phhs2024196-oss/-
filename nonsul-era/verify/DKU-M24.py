#!/usr/bin/env python3
"""DKU-M24 (단국대 2024 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "DKU-M24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 1. g(0)=l, g'(0)=2l/3 ⇒ g=x²+(2l/3)x+l. f 는 f(3)=0 인 monic 이차식 (x-3)(x-p).
    #    f·g≥0 이려면 홀수 중복도의 실근이 없어야 한다. g(3)=9+3l≠0 이므로 3 은 f 에서만 나와 p=3 이어야 하고,
    #    그때 g 는 홀수 중복도 실근이 없어야 한다(판별식 ≤0). 그런 l 에 대해 극값이 모두 정수인지 직접 판정.
    good = []
    for l in range(1, 41):
        g = x**2 + sp.Rational(2, 3) * l * x + l
        assert g.subs(x, 3) != 0
        F = sp.expand((x - 3)**2 * g)
        odd = [r for r, mlt in sp.roots(sp.Poly(F, x)).items() if mlt % 2 == 1 and r.is_real]
        if odd or sp.Poly(F, x).LC() < 0:
            continue                                   # f g ≥ 0 불가(특히 l≥10: g 가 서로 다른 두 실근)
        # p≠3 인 경우 3 이 홀수 중복도 → 불가
        crit = sp.roots(sp.Poly(sp.diff(F, x), x))
        vals = []
        for c, _ in crit.items():
            d = sp.diff(F, x)
            left, right = d.subs(x, c - sp.Rational(1, 10**6)), d.subs(x, c + sp.Rational(1, 10**6))
            if left * right < 0:
                vals.append(sp.nsimplify(F.subs(x, c)))
        if vals and all(v.is_integer for v in vals):
            good.append(l)
    ok1 = len(good) == int(md_answer(1, "1"))
    # 2. f=(x-3)², g=x²+px+q, s>3: f(s)=g(s), f'(s)g'(s)=-1, F'(s)=0 (F=fg), F 는 s 에서 극대
    p, q, s = sp.symbols("p q s", real=True)
    f = (x - 3)**2
    g = x**2 + p * x + q
    sols = sp.solve([f.subs(x, s) - g.subs(x, s),
                     sp.diff(f, x).subs(x, s) * sp.diff(g, x).subs(x, s) + 1,
                     sp.diff(f * g, x).subs(x, s)], [p, q, s], dict=True)
    cand = []
    for so in sols:
        if so[s] <= 3:
            continue
        Fx = sp.expand((f * g).subs({p: so[p], q: so[q]}))
        if Fx.subs(x, so[s]) <= 0:
            continue
        d = sp.diff(Fx, x)
        e = sp.Rational(1, 10**6)
        is_max = d.subs(x, so[s] - e) > 0 > d.subs(x, so[s] + e)
        nonneg = all(not (m % 2 == 1 and r.is_real) for r, m in sp.roots(sp.Poly(Fx, x)).items())
        if is_max and nonneg:
            cand.append((so[s], sp.exp(Fx.subs(x, so[s]))))
    ok2 = len(cand) == 1 and sp.simplify(cand[0][1] - sp.sympify(md_answer(1, "2"))) == 0
    # f 에 g(3)=0 인 경우: f=(x-3)(x-a), g=(x-3)(x-b), s>3 에서 f(s)=g(s) 이면 a=b → f=g → f'g'=f'²≥0≠-1
    # (g(3)=0 경우의 불가능성: f(s)=g(s) ⇒ (s-3)(b-a)=0 ⇒ a=b ⇒ f'g'=f'² ≥ 0 ≠ -1)
    a0 = sp.symbols("a0", real=True)
    imp = sp.solve(sp.diff((x - 3) * (x - a0), x).subs(x, s) ** 2 + 1, a0)
    ok2 = ok2 and imp == []
    # 3. α,β=∓√(M/2a), M=b+c. 사각형 AQBP 의 넓이는 신발끈 공식, ∫_{α/2}^{β/2} S dt = 22a²αβ+33(b+c)
    A, b, c = sp.symbols("A b c", positive=True)
    M = b + c
    al, be = -sp.sqrt(M / (2 * A)), sp.sqrt(M / (2 * A))
    h = lambda u: -A * u**2 + b
    k = lambda u: A * u**2 - c
    P, Q, Aa, Bb = (-t, h(-t)), (t, k(t)), (al, h(al)), (be, h(be))
    pts = [Aa, Q, Bb, P]
    shoelace = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4)) / 2
    # α<t<β 에서의 부호 확인(수치) 후 절댓값 제거
    chk = shoelace.subs({A: 1, b: 30, c: 18, t: 1})
    S = shoelace if chk > 0 else -shoelace
    I = sp.simplify(sp.integrate(S, (t, al / 2, be / 2)))
    eq = sp.simplify(I - (22 * A**2 * al * be + 33 * M))
    # a 별로 M 을 풀어 자연수 (a, b+c) 후보를 구한다: eq=0 ⇔ M=24a(3-a)
    Ms = sp.symbols("Ms", positive=True)
    pairs = []
    for av in range(1, 8):
        r = sp.solve(sp.simplify(eq.subs({b: Ms - c}).subs(A, av)), Ms)
        for m_ in r:
            if m_.is_integer and m_ > 0:
                pairs.append((av, int(m_)))
    vals = set()
    for av, m_ in pairs:
        if -32 * av + m_ > 0:                                  # h(4)>k(4): -16a+b-(16a-c)>0
            for bv in range(1, m_):
                cv = m_ - bv
                hv = lambda u: -av * u**2 + bv
                kv = lambda u: av * u**2 - cv
                vals.add(sum(hv(i) for i in range(3, 12)) - sum(kv(i) for i in range(1, 10)))
    ok3 = len(vals) == 1 and next(iter(vals)) == sp.sympify(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"l={good} G(s)={cand} (a,b+c)={pairs} 합={vals}")


def p02():
    # 1. g(x)g(x+k): x=0, x=-k 에서만 연속성 점검. g(0-)=-5, g(0)=g(0+)=5. 아래는 한쪽 극한을 수치로 직접 계산.
    mp.mp.dps = 40
    fx = lambda u: u**3 - 3 * u**2 - 3 * u + 5
    g = lambda u: -(u**3 - 9 * u + 10) / 2 if u < 0 else fx(u) * mp.e**u
    eps = mp.mpf("1e-25")

    def continuous(k):
        k = mp.mpf(k)
        pts = [mp.mpf(0)] if k == 0 else [mp.mpf(0), -k]
        for z in pts:
            L = g(z - eps) * g(z - eps + k)
            R = g(z + eps) * g(z + eps + k)
            V = g(z) * g(z + k)
            if abs(L - V) > mp.mpf("1e-15") or abs(R - V) > mp.mpf("1e-15"):
                return False
        return True
    Z = set()
    for r in sp.real_roots(sp.Poly(x**3 - 9 * x + 10, x)):
        if r < 0:
            Z.add(r)
    for r in sp.real_roots(sp.Poly(x**3 - 3 * x**2 - 3 * x + 5, x)):
        if r >= 0:
            Z.add(r)
    Zs = {sp.nsimplify(z) for z in Z}
    cand = sorted({0} | {z for z in Zs if sp.nsimplify(-z) in Zs}, key=lambda v: float(v))
    ok_cont = all(continuous(sp.N(v, 40)) for v in cand)
    others = [-2, -1, 1, 2, sp.Rational(1, 2), -3.4, 1.4]
    ok_not = not any(continuous(v) for v in others)
    md = sorted(sp.sympify(md_answer(2, "1").replace("{", "[").replace("}", "]")), key=lambda v: float(v))
    ok1 = ok_cont and ok_not and [sp.simplify(a - b) for a, b in zip(cand, md)] == [0, 0, 0] and len(cand) == len(md)
    # 2. m 자연수: f'=3x²-6x-3m 의 두 실근 a<b 가 -2<a<-1/2, 1<b<3
    ms = []
    for m in range(1, 60):
        r = sorted(sp.real_roots(sp.Poly(3 * x**2 - 6 * x - 3 * m, x)))
        if len(r) == 2 and -2 < r[0] < -sp.Rational(1, 2) and 1 < r[1] < 3:
            ms.append(m)
    m0 = ms[0]
    f = x**3 - 3 * x**2 - 3 * m0 * x + 5
    tot = 0
    for j in range(33):                                       # 단위 구간마다 h(x)=2f(|x-4q|) (주기 4, 우함수)
        mid = j + sp.Rational(1, 2)
        red = ((mid + 2) % 4) - 2
        shift = mid - red
        sg = 1 if red >= 0 else -1
        tot += sp.integrate(2 * f.subs(x, sg * (x - shift)), (x, j, j + 1))
    ok2 = len(ms) == 1 and tot == sp.sympify(md_answer(2, "2"))
    report(2, ok1 and ok2, f"k={cand} m={ms} ∫0^33 h={tot}")


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
