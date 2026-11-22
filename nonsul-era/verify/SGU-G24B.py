#!/usr/bin/env python3
"""SGU-G24B (서강대 2024학년도 수시 논술 자연계열 2차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-G24B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import itertools
    # (1-1) 2x3 격자 최단경로 쌍을 전수: 같은 시각 같은 위치(출발·도착 제외)이면 만남
    paths = []
    for mv in set(itertools.permutations("RRUUU")):
        pos = [(0, 0)]
        for m in mv:
            px, py = pos[-1]
            pos.append((px + 1, py) if m == "R" else (px, py + 1))
        paths.append(pos)
    good = sum(1 for p in paths for q in paths if all(p[i] != q[i] for i in range(1, 5)))
    P1 = sp.Rational(good, len(paths) ** 2)
    ok1 = len(paths) == 10 and P1 == sp.sympify(md_answer(1, "1-1"))
    # (1-2) 다중집합 순열 전수
    items = ["I", "S", "o", "g", "a", "n", "g", "H"]
    arr = set(itertools.permutations(items))
    cnt = sum(1 for w in arr if any(w[i] == "H" and ((i > 0 and w[i - 1] == "g") or (i < 7 and w[i + 1] == "g")) for i in range(8)))
    ok2 = cnt == int(md_answer(1, "1-2"))
    # (1-3) 속력 1 조건에서 f'(t0) 풀이 → 속도
    fp = sp.symbols("fp", real=True)
    F = sp.pi / 3
    sols = sp.solve(sp.Eq(fp**2 + (fp * sp.cos(F))**2, 1), fp)
    vel = {(sp.nsimplify(s), sp.nsimplify(s * sp.cos(F))) for s in sols}
    md3 = {tuple(sp.nsimplify(c) for c in v) for v in sp.sympify(md_answer(1, "1-3"))}
    ok3 = all(any(sp.simplify(a[0] - b[0]) == 0 and sp.simplify(a[1] - b[1]) == 0 for b in vel) for a in md3) and len(md3) == len(vel) == 2
    # (1-4) S(t)=(pi-2t)*2 sin t, S'=0 의 근이 (0,pi/2)에서 유일함을 격자 부호변화로 확인, 적분값
    S = (sp.pi - 2 * t) * 2 * sp.sin(t)
    dS = sp.lambdify(t, sp.diff(S, t))
    import numpy as np
    grid = np.linspace(1e-6, float(sp.pi / 2) - 1e-6, 20001)
    sg = np.sign(dS(grid))
    changes = int(np.sum(sg[1:] != sg[:-1]))
    t0 = sp.nsolve(sp.diff(S, t), t, 0.6, prec=40)
    I = sp.Integral(1 + sp.sec(x)**2, (x, 0, t0)).evalf(30)
    ok4 = changes == 1 and sg[0] > 0 and sg[-1] < 0 and abs(I - sp.N(sp.sympify(md_answer(1, "1-4")), 30)) < 1e-20
    report(1, ok1 and ok2 and ok3 and ok4, f"P={P1} perm={cnt} vel={sorted(vel)} t0={float(t0):.10f} I={float(I):.12f} [증명-수치확인]")


def p02():
    # (2-1) 1보다 큰 실근 개수 [증명-수치확인]
    X = sp.symbols("X", real=True)
    h = X**4 - 8 * X**2 - 1
    roots = [r for r in sp.real_roots(sp.Poly(h, X)) if r > 1]
    ok1 = len(roots) == 1
    a = roots[0]
    # (2-2) 접선·교점·정적분으로 넓이 직접 계산
    T = sp.symbols("T", positive=True)
    l1 = -T**2 * (X - 1 / T) + T          # P1(1/T, T) 에서 접선
    l2 = -1 / T**2 * (X - T) + 1 / T      # P2(T, 1/T) 에서 접선
    b = sp.solve(sp.Eq(l1, l2), X)[0]
    area = sp.integrate(1 / X - l1, (X, 1 / T, b)) + sp.integrate(1 / X - l2, (X, b, T))
    md2 = sp.sympify(md_answer(2, "2-2"), locals={"t": T})
    ok2 = all(abs(float((area - md2).subs(T, v))) < 1e-12 for v in (1.1, 1.7, 2.5, 4.0, 9.0))
    # (2-3) 최솟값 + S(t)<t 격자 확인
    g = T - 2 * sp.log(T)
    cr = sp.solve(sp.diff(g, T), T)
    mn = g.subs(T, cr[0])
    import numpy as np
    Sf = sp.lambdify(T, area)
    ts = np.linspace(1.0001, 60, 6000)
    ok3 = len(cr) == 1 and sp.simplify(mn - sp.sympify(md_answer(2, "2-3"))) == 0 and bool(np.all(Sf(ts) < ts))
    # (2-4) 우극한과 t^2(t^2+1)^3 S''
    p4 = parts(md_answer(2, "2-4"))
    lim = sp.limit(sp.diff(area, T), T, 1, "+")
    E4 = sp.simplify(T**2 * (T**2 + 1)**3 * sp.diff(area, T, 2))
    ok4 = lim == sp.sympify(p4["limit"]) and sp.simplify(E4 - sp.sympify(p4["T"], locals={"t": T})) == 0
    # (2-5) S'>0, S(1+)=0, S->oo, S'' 부호가 t=a 에서 + → - (변곡)
    dS = sp.lambdify(T, sp.diff(area, T)); d2S = sp.lambdify(T, sp.diff(area, T, 2))
    af = float(a)
    ok5 = bool(np.all(dS(ts) > 0)) and sp.limit(area, T, 1, "+") == 0 and sp.limit(area, T, sp.oo) == sp.oo \
        and d2S(af - 0.05) > 0 and d2S(af + 0.05) < 0 \
        and sp.simplify(E4 + 2 * (T**2 - 1) * (T**4 - 8 * T**2 - 1)) == 0
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"root>1={sp.nsimplify(a)}~{af:.6f} b={sp.simplify(b)} min={mn} lim S'={lim} [증명-수치확인]")


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
