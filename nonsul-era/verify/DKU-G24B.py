#!/usr/bin/env python3
"""DKU-G24B (단국대 2024학년도 논술 자연계열 오후) — 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "DKU-G24B"
WIDTH = 2
RESULTS = []
x, t, k, a = sp.symbols("x t k a")
NS = {"pi": sp.pi, "E": sp.E, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"DKU-G24B-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def ans_num(no, sub):
    return float(sp.N(sp.sympify(md_answer(no, sub), locals=NS)))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"DKU-G24B-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def close(u, v, tol=1e-9):
    return abs(float(u) - float(v)) < tol



def p01():
    mp.mp.dps = 25
    s = sp.Symbol("s")
    f = sp.integrate(2 * s / (s**2 + 3), (s, 1, x))
    zer = sp.solveset(sp.Eq(f, 0), x, sp.S.Reals)
    L = sp.log(sp.Rational(4, 3))
    # 삼각형: 밑변=두 영점 사이 거리, 높이=|f(t)|
    base = max(zer) - min(zer)
    ft = f.subs(x, t)
    sols = sp.solveset(sp.Eq(ft, L), t, sp.S.Reals) | sp.solveset(sp.Eq(ft, -L), t, sp.S.Reals)
    yaml_set = sp.FiniteSet(*sp.sympify(md_answer(1, "1"), locals=NS)) if False else sp.FiniteSet(*sp.sympify(md_answer(1, "1"), locals=NS))
    areas_ok = all(abs(sp.N(base * sp.Abs(ft.subs(t, v)) / 2 - L)) < 1e-15 for v in sols)
    ok1 = set(zer) == {-1, 1} and areas_ok and all(sp.simplify(u - v) == 0 for u, v in zip(sorted(sols, key=float), sorted(yaml_set, key=float))) and len(sols) == len(yaml_set)
    # 2. g(x)=ax+8/(x^2+3): 극값 두 개, g(α)>g(β) 인 a 의 범위(격자)
    xs = np.concatenate([np.linspace(-300, -1, 600001), np.linspace(-1, 1, 200001), np.linspace(1, 300, 600001)])

    def good(av):
        gp = av - 16 * xs / (xs**2 + 3)**2
        sg = np.sign(gp)
        idx = np.where(sg[:-1] * sg[1:] < 0)[0]
        if len(idx) != 2: return False
        g = lambda v: av * v + 8 / (v**2 + 3)
        return g(xs[idx[0]]) > g(xs[idx[1]])
    lo, hi = [float(v) for v in re.findall(r"-?\d+\.?\d*", md_answer(1, "2"))][:2]
    ag = [v for v in np.concatenate([np.linspace(-2, 2, 801), [0.01, 0.99, 0.999, 1.001, -0.01]]) if abs(v) > 0.005]
    ok2 = all(good(v) == (lo < v < hi) for v in ag)
    # 3. 정적분은 h 의 선택과 무관: 두 가지 h 가족으로 확인
    al = sp.Symbol("al")
    gx = al * x + 8 / (x**2 + 3)
    h0 = sp.simplify(sp.integrate(2 * s / (s**2 + 3), (s, 1, -1)) - sp.diff(gx, x, 2).subs(x, 0))   # f(-1) - g''(0)
    h1 = sp.simplify(sp.diff(gx, x).subs(x, -1) - gx.subs(x, 1))
    vals = []
    for fam in range(2):
        A_, B_, C_, D_ = sp.symbols("A B C D")
        H = A_ + B_ * x + C_ * x**2 + (D_ * x**3 if fam else 0)
        eqs = [H.subs(x, 0) - h0, H.subs(x, 1) - h1, sp.integrate(sp.diff(H, x) / (x**2 + 3), (x, 0, 1)) + 1]
        unk = [A_, B_, C_] + ([D_] if fam else [])
        so = sp.solve(eqs, unk, dict=True)[0]
        Hs = H.subs(so)
        if fam: Hs = Hs.subs(D_, sp.Rational(1, 3))
        u = sp.Symbol("u")
        val = sp.integrate(sp.sin(2 * x) * Hs.subs(x, sp.cos(x)) / (3 + sp.cos(x)**2)**2, (x, 0, sp.pi / 2)) if False else sp.N(mp.quad(lambda v: mp.sin(2 * v) * sp.lambdify(x, Hs, "mpmath")(mp.cos(v)) / (3 + mp.cos(v)**2)**2, [0, mp.pi / 2]), 20)
        vals.append(float(val))
    ok3 = all(close(v, ans_num(1, "3"), 1e-10) for v in vals) and h0 == sp.Rational(16, 9) and h1 == -1
    report(1, ok1 and ok2 and ok3, f"t={sorted([float(v) for v in sols])} a∈({lo},{hi}) h(0)={h0}, h(1)={h1}, ∫={vals}")


def p02():
    a_ = sp.Symbol("a_")
    A_, b_, c_ = sp.symbols("A_ b_ c_")
    g = (A_ * x**2 + b_ * x + c_) * sp.exp(x)
    P2 = sp.simplify(sp.diff(g, x, 2) / sp.exp(x))
    sol = sp.solve([P2.subs(x, -5), P2.subs(x, 1)], [b_, c_], dict=True)[0]
    ok_f = sp.simplify(sol[b_]) == 0 and sp.simplify(sol[c_] + 7 * A_) == 0
    xs = np.linspace(1e-9, 60, 3000001)

    def feas(av):
        return np.max(av * (xs**2 + 2 * xs - 7) * np.exp(xs)) <= 2 + 1e-12
    lo, hi = -1.0, -1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if feas(mid): hi = mid
        else: lo = mid
    f1 = -6 * hi
    ok1 = ok_f and close(f1, ans_num(2, "1"), 1e-6)
    # 2. k'(x) = h'(h-1)e^h 확인 + 필요성 탐색
    H = sp.Function("H")
    kp = sp.diff(H(x) * sp.exp(H(x)), x) - 2 * sp.exp(H(x)) * sp.diff(H(x), x)
    ok_k = sp.simplify(kp - sp.diff(H(x), x) * (H(x) - 1) * sp.exp(H(x))) == 0
    av = sp.Rational(md_answer(2, "2"))
    hh = av * x * (x - 2)**2 + 1
    Pk = sp.Poly(sp.expand(sp.diff(hh, x) * (hh - 1)), x)
    odd = sorted(r for r, m in sp.roots(Pk).items() if m % 2 == 1)
    r0 = [r for r in odd if r not in (0, 2)]
    ok_ex = odd == [0, sp.Rational(2, 3), 2] and len(r0) == 1 and hh.subs(x, r0[0]) == 3 and hh.subs(x, 2) <= 1 and av > 0
    found = []
    for r in np.arange(-6, 6, 0.0005):
        if abs(r) < 1e-3 or abs(r - 2) < 1e-3: continue
        for c0 in (0, 1):
            for c2 in (0, 1):
                rows = [[r**3, r**2, r, 1], [3 * r**2, 2 * r, 1, 0]]
                rhs = [3, 0]
                rows.append([0, 0, 0, 1] if c0 else [0, 0, 1, 0]); rhs.append(1 if c0 else 0)
                rows.append([8, 4, 2, 1] if c2 else [12, 4, 1, 0]); rhs.append(1 if c2 else 0)
                try:
                    co = np.linalg.solve(np.array(rows, float), np.array(rhs, float))
                except np.linalg.LinAlgError:
                    continue
                if co[0] <= 0 or co[0] > 1e4 or max(abs(co)) > 1e5: continue
                hp = np.poly1d(co); Pp = hp.deriv() * (hp - 1)
                rt = np.roots(Pp)
                rr = sorted(z.real for z in rt if abs(z.imag) < 1e-4)
                groups = []
                for z in rr:
                    if groups and abs(z - groups[-1][0]) < 4e-3: groups[-1][1] += 1
                    else: groups.append([z, 1])
                oddr = [g_[0] for g_ in groups if g_[1] % 2 == 1]
                if len(oddr) == 3 and all(min(abs(o - q) for q in (r, 0, 2)) < 4e-3 for o in oddr) and hp(2) <= 1 + 1e-9:
                    found.append(co[0])
    ok_nec = len(found) > 0 and all(abs(v - 27 / 16) < 0.05 for v in found)
    ok2 = ok_k and ok_ex and ok_nec and close(float(av), ans_num(2, "2"))
    report(2, ok1 and ok2, f"a*={hi:.6f} f(1)max={f1:.6f}; k'식 {ok_k}, 극값점 {odd}, 필요성 탐색 후보 a∈[{min(found):.3f},{max(found):.3f}] ({len(found)}개)" if found else "탐색 후보 없음")


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
