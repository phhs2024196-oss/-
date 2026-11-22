#!/usr/bin/env python3
"""YSU-G26 (연세대 2026학년도 수시 논술 자연계열 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "YSU-G26"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/YSU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 10장 중 5장을 뒤집는 C(10,5)가지를 전수 열거
    tot, cnt = sp.Integer(0), 0
    for T in itertools.combinations(range(1, 11), 5):
        A = set(T) | {0}
        S = sum(A)
        tot += 3 * S if 3 in A else S
        cnt += 1
    E = tot / cnt
    ok = sp.simplify(E - sp.sympify(md_answer(1, "1"))) == 0
    # 대표 오답: 3∈A 여부를 무시하고 E(S)만 구함 = 55/2
    report(1, ok, f"cases={cnt} E(X)={E} (오답 재현 E(S)={sp.Rational(sum(sum(T) for T in itertools.combinations(range(1, 11), 5)), cnt)})")


def p02():
    import mpmath as mp
    nn = sp.symbols("n", positive=True, integer=True)
    # (1) 확률 P(a)=홀수 번 뒤집힐 확률을 이항분포 합으로 직접 계산, 작은 n 에서 전수 경로 열거와 대조
    def p_odd(n):
        q = sp.Rational(1, n)
        return sum(sp.binomial(n, k) * q**k * (1 - q)**(n - k) for k in range(1, n + 1, 2))
    ok_enum = True
    for n in range(1, 7):
        seqs = list(itertools.product(range(n), repeat=n))
        a_enum = sp.Rational(sum(sum(c + 1 for c in range(n) if s.count(c) % 2) for s in seqs), len(seqs))
        ok_enum &= a_enum == sp.Rational(n * (n + 1), 2) * p_odd(n)
    # (2) 큰 n 에서 a_n/n^2 수치 (이항합 직접) → 극한 추정
    mp.mp.dps = 40
    def ratio(n):
        q = mp.mpf(1) / n
        po = mp.fsum(mp.binomial(n, k) * q**k * (1 - q)**(n - k) for k in range(1, n + 1, 2))
        return mp.mpf(n * (n + 1)) / 2 * po / n**2
    r1, r2 = ratio(2000), ratio(4000)
    rich = 2 * r2 - r1                                  # Richardson (오차 O(1/n))
    ans = sp.sympify(md_answer(2, "1"), locals={"e": sp.E})
    ok = ok_enum and abs(rich - mp.mpf(sp.N(ans, 30))) < 1e-6
    report(2, ok, f"enum n<=6 {'ok' if ok_enum else 'NG'}  a_n/n^2(n=4000)={mp.nstr(r2, 10)} richardson={mp.nstr(rich, 12)} ans={sp.N(ans, 12)}")


def nonneg_quad(A, B, C):
    """A x^2 + B x + C >= 0 (모든 실수 x) 판정, 정확한 유리수."""
    if A == 0:
        return B == 0 and C >= 0
    return A > 0 and B * B - 4 * A * C <= 0


def gs(a, b, c):
    return [(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)]   # g_i = p x + q


def feasible(a, b, c):
    # (II) ⟺ 모든 i 에 대해 f-g_i >= 0 (연속함수의 모든 구간 적분이 >=0 ⟺ 피적분함수 >=0)
    if not all(nonneg_quad(a, b - p, c - q) for p, q in gs(a, b, c)):
        return False
    # (I) f(x)>0 인 x 존재
    return a > 0 or b != 0 or c > 0


def p03():
    import random
    R = sp.Rational
    # (3-1) a,b,c ∈ {-6,...,6}/2 전수 → 2f'(26)/f''(26) 값 모음 (a=0 이면 f''=0 이므로 정의되면 a>0)
    grid = [R(k, 2) for k in range(-12, 13)]
    vals, nfeas = set(), 0
    for a, b, c in itertools.product(grid, repeat=3):
        if feasible(a, b, c):
            nfeas += 1
            assert a > 0
            vals.add((2 * (2 * a * 26 + b)) / (2 * a))
    ok1 = nfeas > 0 and vals == {sp.sympify(md_answer(3, "1"))}
    # (II) 동치 근거의 수치 확인: 가능해 하나·불가능해 하나에서 무작위 구간 적분
    random.seed(1)
    def cond2_numeric(a, b, c, trials=400):
        for _ in range(trials):
            s = random.uniform(-5, 5); t = s + random.uniform(1e-3, 3)
            fF = lambda u: a * u**3 / 3 + b * u**2 / 2 + c * u
            for p, q in gs(a, b, c):
                if fF(t) - fF(s) - (p * (t**2 - s**2) / 2 + q * (t - s)) < -1e-9:
                    return False
        return True
    ok_eq = cond2_numeric(26.0, 26.0, 60.0) and not cond2_numeric(26.0, 26.0, 140.0)
    # (3-2) b=26: a ∈ {20,...,32}/... , c 격자 0.5 간격 → f(-1)=a-b+c 의 최소·최대
    f_vals = []
    for a in [R(k, 2) for k in range(40, 65)]:
        for c in [R(k, 2) for k in range(-200, 601)]:
            if feasible(a, R(26), c):
                f_vals.append(a - 26 + c)
    lo, hi = min(f_vals), max(f_vals)
    # 경계 바로 밖이 불가능함도 확인
    ok_edge = not feasible(R(26), R(26), lo - R(1, 1000)) and not feasible(R(26), R(26), hi + R(1, 1000))
    ok2 = ok_edge and lo + hi == sp.sympify(md_answer(3, "2"))
    report(3, ok1 and ok2 and ok_eq,
           f"feasible grid={nfeas} 3-1 values={vals}  3-2 f(-1)∈[{lo},{hi}] sum={lo + hi}  (II)적분수치확인={'ok' if ok_eq else 'NG'}")


def p04():
    import mpmath as mp
    t, k = sp.symbols("t k", real=True)
    # (4-1) g(t)=2026 t + k t^2 (임의 k), f=2 sqrt(1+g^2) 로 C1 위의 점을 두고
    #       Q = P - (곡선 법선 방향 단위벡터) (Q 의 x 좌표가 더 작음)
    g = 2026 * t + k * t**2
    f = 2 * sp.sqrt(1 + g**2)
    gx, gy = f / 2, -2 * g                    # grad(x^2/4 - y^2) = (x/2, -2y), x>0 → 바깥쪽(x 증가) 방향
    nrm = sp.sqrt(gx**2 + gy**2)
    alpha, beta = f - gx / nrm, g - gy / nrm
    assert sp.simplify((alpha - f)**2 + (beta - g)**2 - 1) == 0
    b0 = sp.simplify(sp.diff(beta, t).subs(t, 0))
    ok1 = b0.free_symbols == set() and sp.simplify(b0 - sp.sympify(md_answer(4, "1"))) == 0
    # (4-2) P=(2cosh u, sinh u), 기울기 m(u)=(Qy+√5/2)/Qx 를 u 격자에서 훑어 sup/inf
    mp.mp.dps = 30
    s5 = mp.sqrt(5) / 2
    def slope(u):
        X, Y = 2 * mp.cosh(u), mp.sinh(u)
        ax_, ay_ = X / 2, -2 * Y
        L = mp.sqrt(ax_**2 + ay_**2)
        qx, qy = X - ax_ / L, Y - ay_ / L
        assert qx > 0
        return (qy + s5) / qx
    us = [mp.mpf(i) / 200 for i in range(-4000, 4001)]
    ms = [slope(u) for u in us]
    i_max = max(range(len(ms)), key=lambda i: ms[i])
    umax = mp.findroot(lambda u: mp.diff(slope, u), us[i_max])
    mmax = slope(umax)
    inf_far = slope(mp.mpf(-40))                 # u→-∞ 극한
    minval = min(ms)
    lo_s, hi_s = [s.strip() for s in md_answer(4, "2").strip()[1:-1].split(",")]
    brk = md_answer(4, "2").strip()
    lo, hi = sp.sympify(lo_s), sp.sympify(hi_s)
    ok2 = (abs(mmax - mp.mpf(sp.N(hi, 30))) < 1e-15 and abs(inf_far - mp.mpf(sp.N(lo, 30))) < 1e-15
           and minval > sp.N(lo) and brk[0] == "(" and brk[-1] == "]")
    report(4, ok1 and ok2, f"beta'(0)={b0}  4-2 max={mp.nstr(mmax, 15)} at u={mp.nstr(umax, 8)}, inf(u→-∞)={mp.nstr(inf_far, 15)}, grid min={mp.nstr(minval, 10)} (>inf, 미달성)")


def p05():
    import mpmath as mp
    mp.mp.dps = 30
    f1 = lambda u: (mp.e**u + u) * (1 - u * mp.e**u) / ((mp.e**(2 * u) + 1) * (u**2 + 1))
    f2 = lambda u: (2 * mp.e**u + mp.cos(u)) * (2 - mp.e**u * mp.cos(u)) / ((mp.e**(2 * u) + 1) * (mp.cos(u)**2 + 4))
    xs = [mp.mpf(i) / 100 for i in range(-3000, 3001)]
    i1 = min(range(len(xs)), key=lambda i: f1(xs[i]))
    x1 = mp.findroot(lambda u: mp.diff(f1, u), xs[i1]); m1 = f1(x1)
    i2 = max(range(len(xs)), key=lambda i: f2(xs[i]))
    x2 = mp.findroot(lambda u: mp.diff(f2, u), xs[i2]); m2 = f2(x2)
    a1, a2 = sp.sympify(md_answer(5, "1")), sp.sympify(md_answer(5, "2"))
    ok = abs(m1 - mp.mpf(sp.N(a1, 30))) < 1e-20 and abs(m2 - mp.mpf(sp.N(a2, 30))) < 1e-20
    # 해설의 α: 5-1 은 q(x)=xe^x-e^x-x-1 의 (1,2) 근, 5-2 는 q(x)=e^x cos x+2e^x+cos x-2 의 (-π/2,0) 근에서 극값 달성
    r1 = mp.findroot(lambda u: u * mp.e**u - mp.e**u - u - 1, (1, 2), solver="bisect")
    r2 = mp.findroot(lambda u: mp.e**u * mp.cos(u) + 2 * mp.e**u + mp.cos(u) - 2, (-mp.pi / 2, 0), solver="bisect")
    ok_loc = abs(f1(r1) - m1) < 1e-20 and abs(f2(r2) - m2) < 1e-20
    x1, x2 = r1, r2
    report(5, ok and ok_loc, f"5-1 min={mp.nstr(m1, 20)} at x={mp.nstr(x1, 10)}  5-2 max={mp.nstr(m2, 20)} at x={mp.nstr(x2, 10)}")


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
