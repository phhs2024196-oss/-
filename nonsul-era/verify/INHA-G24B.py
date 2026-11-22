#!/usr/bin/env python3
"""INHA-G24B (2024학년도 인하대 논술 자연계열 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp
from scipy import integrate as si
from scipy import optimize as so

PREFIX = "INHA-G24B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1-1) π^9/9+π^3/3+4/3 (1-2) King 성질 [증명-수치확인] (1-3) 2π-(π/√2)ln(3+2√2)."""
    I1 = sp.integrate((sp.pi - x)**8 + (sp.pi - x)**2 + sp.sin(x)**3, (x, 0, sp.pi))
    ok1 = sp.simplify(I1 - (sp.pi**9 / 9 + sp.pi**3 / 3 + sp.Rational(4, 3))) == 0
    # (1-2) 여러 연속함수 f 에 대해 수치 확인
    ok2 = True
    for f in (lambda s: s**2, lambda s: np.exp(s), lambda s: 1 / (2 + s), lambda s: np.cos(3 * s) + s**5):
        L = si.quad(lambda v: v * f(np.sin(v)), 0, np.pi, epsabs=1e-13)[0]
        R = np.pi / 2 * si.quad(lambda v: f(np.sin(v)), 0, np.pi, epsabs=1e-13)[0]
        ok2 &= abs(L - R) < 1e-10
    # (1-3) 수치 적분과 대조 + sympy 로 반쪽 적분
    num = si.quad(lambda v: (v + 1) * np.sin(v)**3 / (2 - np.cos(v)**2), -np.pi, np.pi, epsabs=1e-13, epsrel=1e-13)[0]
    u = sp.symbols("u", real=True)
    J = sp.pi * sp.integrate((1 - u**2) / (2 - u**2), (u, -1, 1))
    target = 2 * sp.pi - sp.pi / sp.sqrt(2) * sp.log(3 + 2 * sp.sqrt(2))
    ok3 = abs(num - float(target)) < 1e-10 and abs(complex(sp.N(J - target, 30))) < 1e-12
    report(1, ok1 and ok2 and ok3, f"(1-1) {sp.simplify(I1)} (1-3) 수치={num:.12f}, 식={float(target):.12f} [증명-수치확인]")


def p02():
    """(2-1) (-1,1) (2-2) d'(-4)=-3√5/5 (2-3) PQ+PR 최솟값 √5-1."""
    def dist_par(px, py):
        res = so.minimize_scalar(lambda a: (a - px)**2 + (a * a - py)**2, bounds=(-30, 30), method="bounded",
                                 options={"xatol": 1e-13})
        # 전역 확인: 격자
        g = np.linspace(-30, 30, 600001)
        k = np.argmin((g - px)**2 + (g * g - py)**2)
        a0 = so.minimize_scalar(lambda a: (a - px)**2 + (a * a - py)**2, bracket=(g[k] - 1e-3, g[k], g[k] + 1e-3)).x
        return np.sqrt((a0 - px)**2 + (a0 * a0 - py)**2), a0
    d0, a0 = dist_par(-5, -1)
    ok1 = abs(a0 + 1) < 1e-7
    # 기호: 거리 제곱 도함수 인수분해
    s = sp.symbols("s", real=True)
    fd = sp.factor(sp.diff((t + 5)**2 + (t**2 + 1)**2, t))
    # (2-2)
    def d(sv):
        return dist_par(2 * sv + 3, -sv - 6 + np.cos(np.pi * sv))[0]
    h = 1e-5
    dp = (d(-4 + h) - d(-4 - h)) / (2 * h)
    ok2 = abs(dp - float(-3 * sp.sqrt(5) / 5)) < 1e-6 and abs(d(-4) - 2 * np.sqrt(5)) < 1e-9
    # (2-3) 제1사분면 격자에서 dist(P,포물선)+|PS-1| 의 최솟값
    def obj(px, py):
        return dist_par(px, py)[0] + abs(np.hypot(px - 3, py) - 1)
    best = min(obj(px, py) for px in np.linspace(0.05, 4, 40) for py in np.linspace(0.05, 4, 40))
    atP = obj(1.5, 0.75)
    ok3 = abs(atP - (np.sqrt(5) - 1)) < 1e-9 and best >= np.sqrt(5) - 1 - 1e-9
    # 포물선과 S(3,0) 최단거리 = √5
    ds, bs = dist_par(3, 0)
    ok3 &= abs(ds - np.sqrt(5)) < 1e-9 and abs(bs - 1) < 1e-6
    report(2, ok1 and ok2 and ok3, f"(2-1) 최근접 x={a0:.8f}, f'={fd} (2-2) d'(-4)≈{dp:.8f} (2-3) P=(3/2,3/4)에서 {atP:.10f}, 격자 최소 {best:.6f}")


def p03():
    """(3-1)(3-2) 볼록함수 성질 [증명-수치확인] (3-3) p=1/8, q=ln2-3/2."""
    P, Q = sp.symbols("P Q", real=True)
    left, right = P * x**2, x - sp.log(x) + Q
    sol = sp.solve([sp.Eq(left.subs(x, 2), right.subs(x, 2)), sp.Eq(sp.diff(left, x).subs(x, 2), sp.diff(right, x).subs(x, 2))], [P, Q], dict=True)[0]
    pv, qv = sol[P], sol[Q]
    ok3 = pv == sp.Rational(1, 8) and sp.simplify(qv - (sp.log(2) - sp.Rational(3, 2))) == 0
    g2l, g2r = sp.diff(left, x, 2).subs({P: pv, x: 2}), sp.diff(right, x, 2).subs(x, 2)
    ok3 &= g2l == g2r == sp.Rational(1, 4)
    xs = np.linspace(1e-4, 200, 400001)
    gv = np.where(xs <= 2, float(pv) * xs**2, xs - np.log(xs) + float(qv))
    ok3 &= np.all(gv <= xs + 1e-12)
    # (3-1) 무작위 볼록함수에서 할선 기울기 증가
    rng = np.random.default_rng(0)
    ok1 = True
    fs = [lambda v: v**2, lambda v: np.exp(v), lambda v: -np.log(v), lambda v: 1 / v, lambda v: v * np.log(v)]
    for _ in range(3000):
        a_, b_, c_ = np.sort(rng.uniform(0.1, 10, 3))
        if b_ - a_ < 1e-3 or c_ - b_ < 1e-3:
            continue
        f = fs[rng.integers(len(fs))]
        ok1 &= (f(b_) - f(a_)) / (b_ - a_) < (f(c_) - f(b_)) / (c_ - b_)
    # (3-2) 반례 확인: f''>0, f'(α)>1 인 예에서는 f(x)≤x 가 깨짐(f(x)=x^2/8+... 등) — 기울기 k>1 이면 x > (kβ-f(β))/(k-1) 에서 f(x)>x
    f = lambda v: 0.6 * v + 0.05 * v**2 - 3      # f''>0, x>4 에서 f'>1
    beta, alpha = 6.0, 5.0
    k = (f(beta) - f(alpha)) / (beta - alpha)
    X = (k * beta - f(beta)) / (k - 1) + 1e-6
    ok2 = k > 1 and f(X) > X
    # 조건을 만족하는 g 에서는 g' ≤ 1
    gp = np.where(xs <= 2, 2 * float(pv) * xs, 1 - 1 / xs)
    ok2 &= np.all(gp <= 1 + 1e-12)
    report(3, ok1 and ok2 and ok3, f"(3-3) p={pv}, q={qv}, g''(2±)={g2l} [증명-수치확인]")


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
