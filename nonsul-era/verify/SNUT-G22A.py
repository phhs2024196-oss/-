#!/usr/bin/env python3
"""서울과기대 2022학년도 논술전형고사 1차 (SNUT-G22A-01~03) — sympy/mpmath 검산. 정답 하드코딩 비교 없이 문제 조건에서 독립 재계산해 YAML answer 와 비교."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SNUT-G22A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, d, k = sp.symbols("a b d k", real=True)
theta, theta_k, alpha = sp.symbols("theta theta_k alpha", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "d": d, "k": k, "theta": theta,
       "theta_k": theta_k, "alpha": alpha, "pi": sp.pi, "E": sp.E}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d_ = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d_["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def same(u, v, tol=1e-9):
    """수치 동일성(상대오차 tol)."""
    return abs(complex(sp.N(u, 30)) - complex(sp.N(v, 30))) <= tol * max(1, abs(complex(sp.N(v, 30))))

def p01():
    # 1.1 b_n = 1/(1+(n-1)ln(2/3)), a_n = e^(1/b_n) 로 두고 a_n^{b_n}=e 확인, 부분합이 3e 에 수렴
    mp.mp.dps = 40
    bn = lambda m: 1 / (1 + (m - 1) * mp.log(mp.mpf(2) / 3))
    an = lambda m: mp.e ** (1 / bn(m))
    cond = all(abs(an(m) ** bn(m) - mp.e) < mp.mpf(10) ** -30 for m in range(1, 8))
    S = mp.fsum(an(m) for m in range(1, 400))
    ans = load_answer(1, 0)
    report(1, cond and abs(S - sp.N(ans, 30)) < mp.mpf(10) ** -20, f"1.1 sum a_n(400항)={mp.nstr(S, 12)}, yaml={ans}")
    # 1.2 f(0)=0 -> b, 속도 최대 3 -> a, 항등식 계수
    A, B, C, Dd = sp.symbols("A B C D", real=True)
    bs = [s for s in sp.solveset(sp.Eq(2 * sp.cos(B) + 2, 0), B, sp.Interval(0, 2 * sp.pi))]
    assert len(bs) == 1
    b0 = bs[0]
    a0 = sp.Rational(3, 2)                       # v=2a sin(at) 의 최댓값 2a=3
    f = 2 * sp.cos(a0 * t + b0) + 2
    v = sp.diff(f, t)
    vmax = max(sp.N(v.subs(t, tt)) for tt in [sp.pi * j / 60 for j in range(0, 240)])
    sol = sp.solve([(C * (f.subs(t, tt) - 2) ** 2 + 2 * v.subs(t, tt) ** 2 - Dd) for tt in (0, sp.pi / 3)], [C, Dd], dict=True)[0]
    ident = all(abs(sp.N((sol[C] * (f - 2) ** 2 + 2 * v ** 2 - sol[Dd]).subs(t, tt))) < 1e-12 for tt in [0.3, 1.7, 4.2, 9.9, 15.5])
    tot = a0 + b0 + sol[C] + sol[Dd]
    ans = load_answer(1, 1)
    report(1, abs(vmax - 3) < 1e-3 and ident and sp.simplify(tot - ans) == 0, f"1.2 a={a0}, b={b0}, c={sol[C]}, d={sol[Dd]}, 합={tot}, yaml={ans}")
    # 1.3 전수 열거
    cnt = sum(1 for aa in range(1, 15) for bb in range(1, 15) for cc in range(1, 15) for dd in range(1, 15) if aa * (bb + cc + dd) == 14)
    ans = load_answer(1, 2)
    report(1, cnt == ans, f"1.3 순서쌍 {cnt}개, yaml={ans}")


def p02():
    mp.mp.dps = 30
    r = mp.findroot(lambda T: mp.sqrt(64 - T ** 2) - mp.sqrt(25 - T ** 2) - 5, 4.5)
    grid = [mp.mpf(i) / 1013 for i in range(1, 5060)]
    vals = [mp.sqrt(64 - g ** 2) - mp.sqrt(25 - g ** 2) - 5 for g in grid]
    nchg = sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
    ans = load_answer(2, 0)
    report(2, nchg == 1 and abs(r - sp.N(ans, 30)) < mp.mpf(10) ** -20, f"2.1 해 t={mp.nstr(r, 12)}(근 {nchg}개), yaml={ans}")
    tt = sp.Rational(24, 5)
    O = sp.Matrix([0, 0]); A = sp.Matrix([sp.sqrt(25 - tt ** 2), tt]); B = sp.Matrix([sp.sqrt(64 - tt ** 2), tt])
    ang = lambda P, Q, R: sp.acos(((Q - P).dot(R - P)) / (sp.sqrt((Q - P).dot(Q - P)) * sp.sqrt((R - P).dot(R - P))))
    th1 = ang(A, O, B); th2 = ang(B, A, O)
    eq = sp.simplify((A - O).norm() - (B - A).norm()) == 0
    c = sp.simplify(sp.cos(th1 - th2))
    ans = load_answer(2, 1)
    report(2, eq and sp.simplify(c - ans) == 0, f"2.2 OA=AB={sp.simplify((A - O).norm())}, cos(th1-th2)={c}, yaml={ans}")
    a_, b_, c_ = (B - A).norm(), (B - O).norm(), (A - O).norm()
    area = sp.Abs(A[0] * B[1] - A[1] * B[0]) / 2
    R = sp.simplify(a_ * b_ * c_ / (4 * area))
    ans = load_answer(2, 2)
    report(2, sp.simplify(R - ans) == 0, f"2.3 외접원 반지름 R=abc/4K={R}, yaml={ans}")


def p03():
    th = sp.symbols("th", positive=True)
    samples = [0.2, 0.7, 1.1, 1.5]
    Q = sp.Matrix([2 * sp.cos(th), 2 * sp.sin(th)])
    ok = all(abs(sp.N(Q.dot(Q).subs(th, s)) - 4) < 1e-12 and abs(sp.N(sp.atan2(Q[1], Q[0]).subs(th, s)) - s) < 1e-12 for s in samples)
    ans = sp.Matrix(load_answer(3, 0)).subs(theta, th)
    ok = ok and all(abs(sp.N((ans[i] - Q[i]).subs(th, s))) < 1e-12 for i in range(2) for s in samples)
    report(3, ok, "3.1 Q=(2cosθ,2sinθ): 반지름 2, 동경각 θ 확인")
    # 3.2 호 QB 길이 2θ(적분), 접선 방향 단위벡터(시계방향)
    u = sp.symbols("u", positive=True)
    arc = sp.integrate(sp.sqrt(sp.diff(2 * sp.cos(u), u) ** 2 + sp.diff(2 * sp.sin(u), u) ** 2), (u, 0, th))
    T = -sp.diff(Q, th); T = T / sp.sqrt(sp.simplify(T.dot(T)))
    P = sp.simplify(Q + arc * T)
    ans = sp.Matrix(load_answer(3, 1)).subs(theta, th)
    ok = sp.simplify(arc - 2 * th) == 0 and all(abs(sp.N((ans[i] - P[i]).subs(th, s))) < 1e-12 for i in range(2) for s in samples)
    report(3, ok, f"3.2 P={tuple(P)} (실 길이=호 QB={arc})")
    # 3.3 곡선의 길이
    dP = sp.diff(P, th)
    ds = sp.simplify(sp.sqrt(sp.simplify(dP.dot(dP))))
    L1 = sp.integrate(ds, (th, 0, sp.pi / 2))
    ans = load_answer(3, 2)
    report(3, sp.simplify(L1 - ans) == 0, f"3.3 |P'|={ds}, 길이={L1}, yaml={ans}")
    # 3.4 실 전체 길이로부터 세 호
    Ap = sp.Matrix([-2 - 2 * sp.sqrt(3), 0]); Cc = sp.Matrix([-2, 2])
    AC = (Cc - Ap).norm()
    Ltot = AC + 2 + sp.pi                        # A→(-2,2) + (-2,2)→(0,2) + 호 (0,2)→B
    assert sp.simplify(Ltot - (6 + sp.pi)) == 0
    phi = sp.atan2((Cc - Ap)[1], (Cc - Ap)[0])  # 직선 AC 의 기울기각 = π/6
    l2 = (Ltot - AC) * phi                       # (-2,2) 중심, 각 0 → φ
    l3 = Ltot * (sp.pi - phi)                    # A 중심, 각 φ → π
    tot = sp.simplify(L1 + l2 + l3)
    ans = load_answer(3, 3)
    report(3, sp.simplify(phi - sp.pi / 6) == 0 and sp.simplify(tot - ans) == 0, f"3.4 φ={phi}, l2={sp.simplify(l2)}, l3={sp.expand(l3)}, 합={sp.expand(tot)}, yaml={ans}")


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
