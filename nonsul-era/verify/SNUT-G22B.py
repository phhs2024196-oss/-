#!/usr/bin/env python3
"""서울과기대 2022학년도 논술전형고사 2차 (SNUT-G22B-01~03) — sympy/mpmath 검산. 정답 하드코딩 비교 없이 문제 조건에서 독립 재계산해 YAML answer 와 비교."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SNUT-G22B"
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
    # 1.1 접점 조건: 함숫값과 기울기가 모두 같다
    xs, As = sp.symbols("xs As", positive=True)
    A = sp.symbols("A", real=True)
    sol = sp.solve([A * xs - 3 - A * sp.sqrt(xs), A - sp.diff(A * sp.sqrt(x), x).subs(x, xs)], [A, xs], dict=True)
    sol = [s for s in sol if s[A] != 0 and s[xs] > 0]
    assert len(sol) == 1
    x0, a0 = sol[0][xs], sol[0][A]
    y0 = a0 * x0 - 3
    got = f"접점 ({x0}, {y0}), a={a0}"
    ans = load_answer(1, 0, raw=True)
    report(1, got.replace(" ", "") == ans.replace(" ", ""), f"1.1 {got}, yaml={ans}")
    # 1.2 b 의 공비 q: a2*b1+b2*a1=0 -> q=-r
    a1, r, q = sp.symbols("a1 r q", real=True)
    b1 = 1 / a1
    qsol = sp.solve(sp.Eq(a1 * r * b1 + b1 * q * a1, 0), q)[0]
    sols = sp.solve([sp.Eq(b1 / (1 - r), 1), sp.Eq(a1 / (1 - qsol), 2)], [a1, r], dict=True)
    sols = [s for s in sols if abs(sp.N(s[r])) < 1]
    vals = [sp.simplify(s[a1] / (1 - s[r]) + 2 * (1 / s[a1]) / (1 - qsol.subs(r, s[r]))) for s in sols]
    ans = load_answer(1, 1)
    report(1, qsol == -r and len(vals) == 2 and all(sp.simplify(v - ans) == 0 for v in vals), f"1.2 공비 q={qsol}, 해 {len(sols)}개, 합={vals}, yaml={ans}")
    # 1.3
    f = x ** 3 - 6 * x ** 2 + 7
    m = sp.diff(f, x).subs(x, 0)
    line = f.subs(x, 0) + m * x
    xs_ = sorted(sp.solve(sp.Eq(f, line), x))
    area = sp.integrate(sp.Abs(sp.simplify(line - f)).rewrite(sp.Piecewise), (x, 0, 6)) if False else sp.integrate(line - f, (x, xs_[0], xs_[1]))
    below = all(line.subs(x, s) - f.subs(x, s) >= 0 for s in [sp.Rational(i, 4) for i in range(1, 24)])
    ans = load_answer(1, 2)
    report(1, below and sp.simplify(area - ans) == 0, f"1.3 접선 y={line}, 교점 {xs_}, 넓이={area}, yaml={ans}")


def p02():
    aa, bb = sp.symbols("aa bb", positive=True)
    c = sp.symbols("c", positive=True)
    # 2.1 접선 기울기 = 직선 AB 기울기
    cs = sp.solve(sp.Eq(sp.diff(sp.log(x), x).subs(x, c), (sp.log(bb) - sp.log(aa)) / (bb - aa)), c)[0]
    ans = load_answer(2, 0).subs({a: aa, b: bb})
    report(2, sp.simplify(cs - ans) == 0, f"2.1 C 의 x좌표={cs}, yaml={ans}")
    # 2.2 곡선과 꺾은선 A-C1-B 사이의 넓이를 수치적분으로 직접 계산
    mp.mp.dps = 25
    ok = True
    for av, bv in [(0.1, 0.5), (0.2, 0.9), (0.05, 0.3), (0.4, 0.8), (0.3, 0.6)]:
        av, bv = mp.mpf(av), mp.mpf(bv)
        cv = (bv - av) / (mp.log(bv) - mp.log(av))
        la = lambda X: mp.log(av) * (cv - X) / (cv - av)      # A(av,ln av)→C1(cv,0)
        lb = lambda X: mp.log(bv) * (X - cv) / (bv - cv)      # C1→B(bv,ln bv)
        S = mp.quad(lambda X: la(X) - mp.log(X), [av, cv]) + mp.quad(lambda X: lb(X) - mp.log(X), [cv, bv])
        form = sp.N(load_answer(2, 1).subs({a: sp.Float(str(av), 25), b: sp.Float(str(bv), 25)}), 25)
        ok = ok and S > 0 and abs(S - mp.mpf(str(form))) < mp.mpf(10) ** -15
    report(2, ok, "2.2 꺾은선-곡선 사이 넓이(수치적분) = yaml 식 (5개 (a,b) 표본)")
    # 2.3 b=2a 최댓값
    Sa = load_answer(2, 1).subs({a: aa, b: 2 * aa})
    crit = sp.solve(sp.diff(Sa, aa), aa)
    crit = [s for s in crit if 0 < s < sp.Rational(1, 2)]
    val = sp.simplify(Sa.subs(aa, crit[0]))
    num = max(sp.N(Sa.subs(aa, sp.Rational(i, 1000))) for i in range(1, 500))
    ans = load_answer(2, 2)
    report(2, len(crit) == 1 and sp.simplify(val - ans) == 0 and abs(num - sp.N(ans)) < 1e-4, f"2.3 임계점 a={crit}, 최댓값={val}, 격자 최대≈{num:.6f}, yaml={ans}")


def p03():
    # 3.1 좌표: O(0,0), A(1,d-x), B(3,d)
    dd, xx, kk = sp.symbols("dd xx kk", positive=True)
    O = sp.Matrix([0, 0]); A = sp.Matrix([1, dd - xx]); B = sp.Matrix([3, dd])
    fx = (A - O).norm() + kk * (B - A).norm()
    ans = load_answer(3, 0).subs({d: dd, x: xx, k: kk})
    report(3, sp.simplify(fx - ans) == 0, f"3.1 f(x)={sp.simplify(fx)}")
    # 3.2 k=3/2, x=√2 에서 f'=0 → d, θ1=atan(d-x)
    df = sp.diff(fx, xx).subs({kk: sp.Rational(3, 2), xx: sp.sqrt(2)})
    ds = [s for s in sp.solve(df, dd) if sp.N(s) > 0]
    d0 = ds[0]
    f0 = fx.subs({kk: sp.Rational(3, 2), dd: d0})
    mn = all(sp.N(f0.subs(xx, s)) >= sp.N(f0.subs(xx, sp.sqrt(2))) - 1e-12 for s in [sp.Rational(i, 20) for i in range(1, 60)])
    th1 = sp.atan((d0 - sp.sqrt(2)) / 1)
    ans = load_answer(3, 1)
    report(3, len(ds) == 1 and mn and sp.simplify(th1 - ans) == 0, f"3.2 d={d0}, 격자에서 x=√2 가 최소={mn}, θ1={sp.simplify(th1)}, yaml={ans}")
    # 3.3 B 에서 직선 OA 까지 거리
    Bv = sp.Matrix([3, d0]); Av = sp.Matrix([1, d0 - sp.sqrt(2)]); Ov = sp.Matrix([0, 0])
    u = Av - Ov; w = Bv - Ov
    h = sp.simplify(sp.Abs(u[0] * w[1] - u[1] * w[0]) / u.norm())
    ans = load_answer(3, 2)
    report(3, sp.simplify(h - ans) == 0, f"3.3 h={h}, yaml={ans}")


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
