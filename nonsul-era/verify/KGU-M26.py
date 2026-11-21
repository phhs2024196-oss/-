#!/usr/bin/env python3
"""KGU-M26 (경기대 2026학년도 모의논술 수리 문제 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer 와 비교."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KGU-M26"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
a = sp.symbols("a", positive=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub="1"):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KGU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    sp_ = next(s for s in d["subparts"] if str(s["id"]) == str(sub))
    return sp.sympify(sp_["answer"]["value"], locals={"a": a, "t": t})


def _area(P):
    """다각형 넓이(꼭짓점 순서대로)."""
    s = 0
    for i in range(len(P)):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % len(P)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def p01():
    """(1) 포물선을 세 조건으로 결정, 사다리꼴(AB∥CD) 넓이를 t 의 함수로 최대화(구간 끝값 포함).
    추가로 AB∥CD 를 가정하지 않은 사각형 ABCD(C,D 포물선 위, y>0) 전체를 격자 탐색해 최댓점이 같은지 확인.
    (2) CD 지름 원과 y축의 두 교점 E 각각에서 CE·DE 를 직접 계산."""
    p, q, r = sp.symbols("p q r")
    g = p * x**2 + q * x + r
    sol = sp.solve([g.subs(x, -a), g.subs(x, a), g.subs(x, 0) - a**2], [p, q, r], dict=True)[0]
    G = g.subs(sol)
    S = (2 * a + 2 * t) / 2 * G.subs(x, t)          # C(t,G(t)), D(-t,G(t))
    crit = [c for c in sp.solve(sp.diff(S, t), t) if (c > 0) == True and (c < a) == True]
    cand = crit + [sp.Integer(0), a]
    tmax = max(cand, key=lambda c: S.subs(t, c).subs(a, 3))
    C = (tmax, sp.simplify(G.subs(x, tmax)))
    D = (-tmax, sp.simplify(G.subs(x, -tmax)))
    ans1 = load_answer(1, "1")
    ok1 = all(sp.simplify(ans1[0][i] - C[i]) == 0 and sp.simplify(ans1[1][i] - D[i]) == 0 for i in range(2))
    # 일반 사각형 격자 탐색 (a=1.7): D(xd), C(xc), -a<xd<xc<a
    A0 = 1.7
    Gf = sp.lambdify(x, G.subs(a, A0))
    N = 600
    xs = [-A0 + 2 * A0 * k / N for k in range(1, N)]
    best = max(((_area([(-A0, 0), (A0, 0), (xc, Gf(xc)), (xd, Gf(xd))]), xd, xc)
                for xd, xc in itertools.combinations(xs, 2)), key=lambda z: z[0])
    ok_grid = abs(best[1] + A0 / 3) < 2 * 2 * A0 / N + 1e-9 and abs(best[2] - A0 / 3) < 2 * 2 * A0 / N + 1e-9
    # (2)
    cx, cy = (C[0] + D[0]) / 2, (C[1] + D[1]) / 2
    R = sp.sqrt((C[0] - cx) ** 2 + (C[1] - cy) ** 2)
    yE = sp.solve(sp.Eq((0 - cx) ** 2 + (x - cy) ** 2, R**2), x)
    prods = set()
    for ye in yE:
        E = (0, ye)
        CE = sp.sqrt((C[0] - E[0]) ** 2 + (C[1] - E[1]) ** 2)
        DE = sp.sqrt((D[0] - E[0]) ** 2 + (D[1] - E[1]) ** 2)
        prods.add(sp.simplify(CE * DE))
    ans2 = load_answer(1, "2")
    ok2 = len(prods) == 1 and sp.simplify(prods.pop() - ans2) == 0 and len(yE) == 2
    report(1, ok1 and ok_grid and ok2,
           f"포물선 y={G}, (1) C={C}, D={D}, md={ans1}; 일반 사각형 격자 최댓점 xd={best[1]:.4f}, xc={best[2]:.4f} (±a/3={A0/3:.4f}); "
           f"(2) E 두 점 모두 곱 일치, md={ans2}")


def p02():
    """(1) v=t^3+p t^2+q t 로 두고 c),d) 로부터 t=1 에서 연속·좌우미분계수 일치를 기호로 세워 풂.
    결과로 만든 구간별 v 가 t=1..9 에서 미분가능한지 수치 확인.
    (2) 구간별 v(t)=(-1/2)^n v(t-n) 을 직접 적분(|v|) 하여 0~10 의 거리 합."""
    pp, qq, h = sp.symbols("p q h", real=True)
    v0 = t**3 + pp * t**2 + qq * t                      # a), b)
    # t>1 에서 v(t) = -1/2 v(t-1)  (t-1 in (0,1])
    right = lambda tt: -sp.Rational(1, 2) * v0.subs(t, tt - 1)
    cont = sp.Eq(v0.subs(t, 1), sp.limit(right(1 + h), h, 0, "+"))
    qsol = sp.solve(cont, qq)                            # 연속 조건 먼저 적용(미분가능 ⇒ 연속)
    assert len(qsol) == 1
    w0 = v0.subs(qq, qsol[0])
    dl = sp.limit((w0.subs(t, 1 + h) - w0.subs(t, 1)) / h, h, 0, "-")
    dr = sp.limit((-sp.Rational(1, 2) * w0.subs(t, h) - w0.subs(t, 1)) / h, h, 0, "+")
    psol = sp.solve(sp.Eq(dl, dr), pp)
    assert len(psol) == 1
    sol = [{pp: psol[0], qq: qsol[0].subs(pp, psol[0])}]
    V = sp.expand(v0.subs(sol[0]))
    ans1 = load_answer(2, "1")
    ok1 = sp.simplify(V - ans1) == 0

    def vfun(n):  # [n, n+1] 의 식
        return sp.expand((-sp.Rational(1, 2)) ** n * V.subs(t, t - n))
    diff_ok = all(
        sp.simplify(vfun(k - 1).subs(t, k) - vfun(k).subs(t, k)) == 0
        and sp.simplify(sp.diff(vfun(k - 1), t).subs(t, k) - sp.diff(vfun(k), t).subs(t, k)) == 0
        for k in range(1, 10))
    dist = 0
    for k in range(10):
        f = vfun(k)
        roots = sorted([r for r in sp.solve(f, t) if r.is_real and k < r < k + 1])
        pts = [sp.Integer(k)] + roots + [sp.Integer(k + 1)]
        dist += sum(sp.Abs(sp.integrate(f, (t, pts[i], pts[i + 1]))) for i in range(len(pts) - 1))
    dist = sp.nsimplify(dist)
    ans2 = load_answer(2, "2")
    # 대표 오답 재현: |v| 대신 v 를 그대로 적분(변위)
    disp = sp.nsimplify(sum(sp.integrate(vfun(k), (t, k, k + 1)) for k in range(10)))
    ok2 = dist == ans2
    report(2, ok1 and ok2 and diff_ok,
           f"(1) v={V}, md={ans1}, t=1..9 이음 미분가능={diff_ok} / (2) 거리={dist}, md={ans2} (오답: 변위={disp})")


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
