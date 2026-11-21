#!/usr/bin/env python3
"""SJU-G19A (세종대 2019학년도 수시 논술고사[A]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "SJU-G19A"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)
E = sp.E


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str) -> dict:
    """'키=식; 키=식' 형태의 정답을 dict 로."""
    return {a.strip(): b.strip() for a, b in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def S(s, **loc):
    return sp.sympify(s, locals={"e": sp.E, "x": x, "t": t, **loc})


def same(a, b, tol=1e-12):
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 if not (isinstance(a, float) or isinstance(b, float)) else abs(float(a) - float(b)) < tol
    except Exception:  # noqa: BLE001
        return abs(float(a) - float(b)) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    ph = sp.Rational(24, 72)
    se = sp.sqrt(ph * (1 - ph) / 72)
    s0, t0 = ph - sp.Rational(196, 100) * se, ph + sp.Rational(196, 100) * se
    d = md_parts(1, "1")
    ok1 = sp.simplify(S(d["s"]) - s0) == 0 and sp.simplify(S(d["t"]) - t0) == 0
    pmf_md = md_answer(1, "2").split(" (")[0]
    kk, nn = sp.symbols("k n", integer=True)
    ok2 = ok3 = True
    for N in range(4, 11):
        cnt = {}
        for c in itertools.combinations(range(1, N + 1), 4):
            cnt[max(c)] = cnt.get(max(c), 0) + 1
        tot = math.comb(N, 4)
        for K in range(4, N + 1):
            ok2 &= sp.Rational(cnt.get(K, 0), tot) == S(pmf_md, k=kk, n=nn).subs({kk: K, nn: N})
        a = sum(sp.Rational(K * v, tot) for K, v in cnt.items())
        cnt2 = {}
        for c in itertools.combinations(range(1, N + 2), 4):
            cnt2[max(c)] = cnt2.get(max(c), 0) + 1
        b = sum(sp.Rational(K * v, math.comb(N + 1, 4)) for K, v in cnt2.items())
        A = sp.Symbol("a")
        ok3 &= S(md_answer(1, "3"), a=A).subs(A, a) == b
        ok3 &= (sp.Rational(N - 3, N + 1) * a + 4) == b          # 원 해설 형태도 성립
    report(1, ok1 and ok2 and ok3, f"s={s0} t={t0} pmf(n=4..10 전수) b=a+4/5 확인")


def p02():
    tt, ss = sp.symbols("tt ss", real=True)
    A = sp.Matrix([1, 1, 2]); n1 = sp.Matrix([-1, 1, 1]); n2 = sp.Matrix([1, -1, 1])
    C = A + tt * n1
    sols = sp.solve(sp.Eq((C - A).dot(C - A), (C.dot(n2) - 4)**2 / 3), tt)
    Cs = [C.subs(tt, v) for v in sols if C.subs(tt, v)[0] > 0]
    ok1 = len(Cs) == 1 and sp.Matrix(S(md_answer(2, "1"))) == Cs[0]
    C0 = Cs[0]; r = sp.sqrt((C0 - A).dot(C0 - A))
    Bp = C0 + ss * n2; sv = sp.solve(sp.Eq(Bp.dot(n2), 4), ss)[0]; B = Bp.subs(ss, sv)
    ok2 = sp.Matrix(S(md_answer(2, "2"))) == B
    import numpy as np
    c = np.array([float(v) for v in C0]); a = np.array([float(v) for v in A]); b = np.array([float(v) for v in B])
    best, bp = -1e9, None
    for th in np.linspace(0, np.pi, 361):
        for ph in np.linspace(0, 2 * np.pi, 721):
            P = c + float(r) * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            v = (a - P) @ (b - P)
            if v > best:
                best, bp = v, P
    Pm = np.array([float(v) for v in S(md_answer(2, "3"))])
    vm = (a - Pm) @ (b - Pm)
    ok3 = abs(np.linalg.norm(Pm - c) - float(r)) < 1e-12 and vm >= best - 1e-9 and np.linalg.norm(bp - Pm) < 1e-2
    report(2, ok1 and ok2 and ok3, f"C={list(C0)} B={list(B)} max PA.PB={vm:.6f} (격자 {best:.6f})")


def p03():
    a = sp.Symbol("a", positive=True); y = sp.Symbol("y", positive=True)
    ok1 = True
    for N in range(1, 7):
        f = y**N * (a + sp.log(y))
        Fe = sp.integrate(f, (y, 1, sp.E))
        sol = sp.solve(sp.Eq(Fe, (a * N + N + a) * sp.E**(N + 1) / (N + 1)**2), a)
        ok1 &= len(sol) == 1 and sp.simplify(sol[0] - S(md_answer(3, "1"), n=sp.Integer(N))) == 0
    import numpy as np
    good = []
    for N in range(1, 9):
        av = 1 / (N + 1)
        xs = np.linspace(1, 60, 20001)
        fp = (N * np.log(xs) + av * N + 1) * xs**(N - 1)
        if np.all(fp >= 2 * av + 1 - 1e-12):
            good.append(N)
    nn = sp.Symbol("n")
    rel = S(md_answer(3, "2"), n=nn)
    ok2 = good == [N for N in range(1, 9) if bool(rel.subs(nn, N))]
    vals = []
    for N in good:
        f = y**N * (sp.Rational(1, N + 1) + sp.log(y))
        vals.append(sp.diff(f, y).subs(y, 1))
    ok3 = sp.simplify(min(vals) - S(md_answer(3, "3"))) == 0
    Ns = [N for N in range(2, 12) if sp.simplify(sp.integrate(y**N * (sp.Rational(1, N + 1) + sp.log(y)), (y, 1, sp.E)) - sp.E**4 / 4) == 0]
    N = Ns[0]
    f = y**N * (sp.Rational(1, N + 1) + sp.log(y)); Fe = sp.integrate(f, (y, 1, sp.E))
    m = f.subs(y, sp.E); b = sp.simplify(Fe - m * sp.E)
    d = md_parts(3, "4")
    ok4 = len(Ns) == 1 and sp.simplify(S(d["m"]) - m) == 0 and sp.simplify(S(d["b"]) - b) == 0
    Fl = sp.lambdify(y, sp.integrate(f, (y, 1, y)))
    xs = np.linspace(1, 8, 3001)
    ok4 &= all(float(m) * xx + float(b) <= Fl(xx) + 1e-9 for xx in xs)
    report(3, ok1 and ok2 and ok3 and ok4, f"a=1/(n+1) n:{good[:3]}.. minF''(1)={min(vals)} n={N} m={m} b={b} [증명-수치확인]")


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
