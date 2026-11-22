#!/usr/bin/env python3
"""한국외대 2022학년도 논술전형(자연계) 1~10 (HUFS-G22-01~10) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HUFS-G22"
WIDTH = 2
RESULTS = []
x, t, n, r = sp.symbols("x t n r", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "r": r, "a": a, "b": b, "k": k}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/HUFS/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """x^2-3x+1=0 의 두 근 각각을 tanθ 로 놓고 θ=atan(근) ∈ (0, π/2) 에서 sinθcosθ 직접 계산."""
    roots = sp.solve(x**2 - 3 * x + 1, x)
    vals = []
    for rt in roots:
        th = sp.atan(rt)
        assert 0 < float(th) < float(sp.pi / 2)
        vals.append(sp.nsimplify(sp.simplify(sp.sin(th) * sp.cos(th))))
    ans = load_answer(1)
    ok = all(sp.simplify(v - ans) == 0 for v in vals)
    report(1, ok, f"근 {roots} 각각 sinθcosθ={vals}, yaml={ans}")


def p02():
    """교점을 solve 로 구하고, 인접 교점 사이 구간마다 ∫|f-g| 를 직접 적분(부호는 구간 중점에서 판정)."""
    f, g = x**3 - x, 3 * x
    xs = sorted(sp.solve(f - g, x))
    S = 0
    for lo, hi in zip(xs, xs[1:]):
        mid = (lo + hi) / 2
        d = f - g if (f - g).subs(x, mid) > 0 else g - f
        S += sp.integrate(d, (x, lo, hi))
    ans = load_answer(2)
    report(2, sp.simplify(S - ans) == 0, f"교점 {xs}, 넓이={S}, yaml={ans} (오답 재현: 부호 무시 ∫(f-g)={sp.integrate(f-g,(x,xs[0],xs[-1]))})")


def p03():
    """원 위의 점 (r cos s, r sin s) 와 직선 rx+y-2r=0 의 거리 = |r^2 cos s + r sin s - 2r|/sqrt(r^2+1).
    합성(진폭 R=sqrt((r^2)^2+r^2))으로 최대·최소를 얻고, 극한은 sympy limit. 수치: s 격자 20000점으로 r=1e-3 비 확인,
    r∈(0,√3) 격자에서 최소거리>0(직선이 원과 안 만남) 확인."""
    import math
    rp = sp.symbols("rp", positive=True)
    R = sp.sqrt((rp**2)**2 + rp**2)
    den = sp.sqrt(rp**2 + 1)
    Mx = (2 * rp + R) / den
    mn = (2 * rp - R) / den
    L = sp.limit(Mx / mn, rp, 0, "+")

    def num(rv):
        ds = [abs(rv * rv * math.cos(2 * math.pi * i / 20000) + rv * math.sin(2 * math.pi * i / 20000) - 2 * rv) / math.sqrt(rv * rv + 1) for i in range(20000)]
        return max(ds), min(ds)
    M1, m1 = num(1e-3)
    ok_num = abs(M1 - float(Mx.subs(rp, 1e-3))) < 1e-9 and abs(m1 - float(mn.subs(rp, 1e-3))) < 1e-9
    ok_out = all(num(0.0173 * i)[1] > 0 for i in range(1, 100))
    ans = load_answer(3)
    ok = sp.simplify(L - ans) == 0 and ok_num and ok_out
    report(3, ok, f"M={sp.simplify(Mx)}, m={sp.simplify(mn)}, 극한={L}, 격자 비(r=1e-3)={M1/m1:.6f}, 원 밖 {ok_out}, yaml={ans}")


def p04():
    """[증명-수치확인] n=1..60 에서 pow(9, 2^n, 2^(n+3)) == 1 직접 확인 + 귀납 단계 항등식 sympy 전개 확인."""
    ok_num = all(pow(9, 2**m, 2**(m + 3)) == 1 for m in range(1, 61))
    q, kk = sp.symbols("q k", integer=True, nonnegative=True)
    lhs = sp.expand((2**(kk + 3) * q + 1)**2)
    rhs = sp.expand(2**(kk + 4) * (2**(kk + 2) * q**2 + q) + 1)
    ok_id = sp.simplify(sp.powsimp(lhs - rhs, force=True)) == 0
    raw = load_answer(4, raw=True)
    report(4, ok_num and ok_id and raw is None, f"[증명-수치확인] n=1..60 나머지 1: {ok_num}, (2^(k+3)q+1)^2 = 2^(k+4)(2^(k+2)q^2+q)+1: {ok_id}, yaml 증명형 value={raw}")


def p05():
    """(나)(다)를 연립해 a,b,c 를 solve, 제약 log3a·log_m b·log_n c=1 에서 u=log3 m 에 대한 log3 mn 을 미분으로 최소화."""
    kk = sp.symbols("kk", positive=True)
    ksol = sp.solve(sp.Eq(sp.log(kk**3 * kk**5 / kk**2, 3), 3), kk)[0]
    A, B, C = ksol**2, ksol**3, ksol**5
    u, v = sp.symbols("u v", positive=True)   # u=log3 m, v=log3 n (m,n>1 → 양수)
    vsol = sp.solve(sp.Eq(sp.log(A, 3) * (sp.log(B, 3) / u) * (sp.log(C, 3) / v), 1), v)[0]
    F = u + vsol
    cu = [c for c in sp.solve(sp.diff(F, u), u) if c.is_positive]
    u0 = cu[0]
    got = (sp.simplify(F.subs(u, u0)), sp.simplify(u0), sp.simplify(vsol.subs(u, u0)))
    ans = load_answer(5)
    ok = all(sp.simplify(g - w) == 0 for g, w in zip(got, ans)) and sp.limit(F, u, 0, "+") == sp.oo and sp.limit(F, u, sp.oo) == sp.oo
    report(5, ok, f"a,b,c={A},{B},{C}, uv={sp.simplify(u*vsol)}, (최솟값, log3m, log3n)={got}, yaml={ans}")


def p06():
    """A(α, log3 9α), C(α, log3 α), B(u, log3 9u), u<α 를 미지수로 두고 |AB|^2=|BC|^2=|AC|^2 를 nsolve(여러 초기값) → 해 유일성 확인 후 α·3^β."""
    import mpmath as mp
    mp.mp.dps = 40
    L3 = lambda z: mp.log(z) / mp.log(3)

    def eqs(al, u):
        A = (al, L3(9 * al)); C = (al, L3(al)); B = (u, L3(9 * u))
        d2 = lambda P, Q: (P[0] - Q[0])**2 + (P[1] - Q[1])**2
        return [d2(A, B) - d2(A, C), d2(B, C) - d2(A, C)]
    sols = set()
    for a0 in [0.5, 1, 2, 3, 5, 8]:
        for du in [0.5, 1.5, 2.5]:
            try:
                s_ = mp.findroot(lambda al, u: eqs(al, u), (a0, max(a0 - du, 0.05)))
                al, u = s_[0], s_[1]
                if al > 0 and 0 < u < al:
                    sols.add((mp.nstr(al, 25), mp.nstr(u, 25)))
            except Exception:  # noqa: BLE001
                pass
    vals = {mp.nstr(mp.mpf(al) * 3**L3(mp.mpf(al)), 25) for al, _ in sols}
    ans = load_answer(6)
    ok = len(vals) == 1 and abs(mp.mpf(vals.pop()) - mp.mpf(str(sp.N(ans, 40)))) < mp.mpf("1e-20")
    report(6, ok, f"해 (α, B의 x)={sorted(sols)}, α·3^β 수치 일치 {ok}, yaml={ans} (α 정확값 3√3/2≈{float(3*sp.sqrt(3)/2):.10f})")


def p07():
    """접선을 diff 로 구하고 교점 방정식을 solve, P(x=-1) 제외한 근들의 제곱합 계산."""
    f = x**4 + 3 * x**3
    assert f.subs(x, -1) == -2
    tl = sp.diff(f, x).subs(x, -1) * (x + 1) + f.subs(x, -1)
    rts = [s_ for s_ in sp.solve(f - tl, x) if s_ != -1]
    S = sp.simplify(sum(s_**2 for s_ in rts))
    ans = load_answer(7)
    report(7, len(rts) == 2 and sp.simplify(S - ans) == 0, f"접선 y={sp.expand(tl)}, Q·R x좌표 {rts}, 제곱합={S}, yaml={ans}")


def p08():
    """S_n=pn^2+20n 에서 a_1..a_8 을 직접 (S_n - S_{n-1}, a_1=S_1) 만들고 (나)로 p 를 solve, 짝수항 합 계산(채점 유의사항 3 방식)."""
    p = sp.symbols("p")
    S = lambda m: p * m**2 + 20 * m
    A = {m: S(m) - (S(m - 1) if m > 1 else 0) for m in range(1, 9)}
    ps = sp.solve(sp.Eq(A[1] + A[3] + A[5] + A[7], 24), p)
    val = sp.simplify((A[2] + A[4] + A[6] + A[8]).subs(p, ps[0]))
    ans = load_answer(8)
    report(8, len(ps) == 1 and sp.simplify(val - ans) == 0, f"p={ps}, a_1..a_8={[A[m].subs(p, ps[0]) for m in range(1, 9)]}, 짝수항 합={val}, yaml={ans}")


def p09():
    """신발끈 공식으로 넓이 T(t)=|det|/2 를 직접 세우고, (0,3) 에서 미분해 최대 + 격자 3000점 확인."""
    f = x**3 - 3 * x**2 - 4 * x
    assert f.subs(x, 3) == -12
    det = 3 * f.subs(x, t) - (-12) * t
    T = sp.Abs(det) / 2
    Tin = -det / 2 if det.subs(t, 1) < 0 else det / 2          # (0,3) 에서 부호 일정
    sign_ok = all((det.subs(t, sp.Rational(i, 100)) * det.subs(t, 1)) > 0 for i in range(1, 300))
    crit = [c for c in sp.solve(sp.diff(Tin, t), t) if 0 < c < 3]
    mx = max(Tin.subs(t, c) for c in crit)
    grid = max(float(T.subs(t, 3 * i / 3000)) for i in range(1, 3000))
    ans = load_answer(9)
    ok = sign_ok and sp.simplify(mx - ans) == 0 and abs(grid - float(ans)) < 1e-6
    report(9, ok, f"넓이(t)={sp.factor(Tin)}, 임계점 {crit}, 최댓값={mx}, 격자≈{grid:.6f}, yaml={ans}")


def p10():
    """시뮬레이션: k차 접종 시점 k-1, 총 면역력 I(s)=Σ_{접종시점 τ≤s} A·(1/2)^{s-τ} (A=1).
    시간 격자(0.001년)로 0~30년을 훑어, 'n회 접종 이후(시점 n-1 부터) 계속 ≥0.8' 인 최소 n 을 직접 찾는다."""
    from fractions import Fraction as Fr
    T, step = 30, 1000

    def I(s):  # s: Fraction 시간
        return sum(0.5 ** float(s - tau) for tau in range(0, int(s) + 1))
    vals = [I(Fr(i, step)) for i in range(0, T * step)]

    def ok_from(nv):  # n 회 접종 직후 시점 = n-1
        return all(v >= 0.8 - 1e-12 for v in vals[(nv - 1) * step:])
    nmin = next(nv for nv in range(1, T) if ok_from(nv))
    # 닫힌식 확인: a_n = 1-(1/2)^{n-1} 이 접종 직전 값과 일치
    an = [sp.Rational(1) - sp.Rational(1, 2)**(m - 1) for m in range(1, 8)]
    pre = [I(Fr(m - 1) - Fr(1, 10**9)) if m > 1 else 0.0 for m in range(1, 8)]
    close_ok = all(abs(float(a_) - p_) < 1e-6 for a_, p_ in zip(an, pre))
    ans = load_answer(10)
    report(10, nmin == ans and close_ok, f"시뮬레이션 최소 접종 횟수={nmin}, a_1..a_7={an} (직전값 일치 {close_ok}), yaml={ans} (오답 재현: a_n≥0.8 로 쓰면 n={next(m for m in range(1,20) if an[m-1] >= sp.Rational(4,5)) if True else ''})")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY G22: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
