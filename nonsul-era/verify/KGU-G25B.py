#!/usr/bin/env python3
"""KGU-G25B (경기대 2025학년도 논술우수자전형 수리 B형 문항 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "KGU-G25B"
WIDTH = 2
RESULTS = []
x, n = sp.symbols("x n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub="1", local=None):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KGU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    sp_ = next(s for s in d["subparts"] if str(s["id"]) == str(sub))
    return sp.sympify(sp_["answer"]["value"], locals=local or {})


def p01():
    """B 지점을 원점, A 지점을 x=1 로 두고(AB=1) 각 파편의 직선을 두 높이 점으로 세운 뒤
    (1) 지면과 이루는 각의 tan = |기울기| 합, (2) 지면 도달점과 B 의 거리 합을 직접 계산.
    좌표 방향을 뒤집은 배치(A=0, B=1)로도 같은 값인지 확인."""
    tans, dists, tans2, dists2 = [], [], [], []
    for k in range(1, 9):
        hA, hB = sp.Integer(k + 1) ** 2, sp.Integer(1)
        # 배치 1: A(1,hA), B(0,hB)
        line = sp.Line(sp.Point(1, hA), sp.Point(0, hB))
        land = line.intersection(sp.Line(sp.Point(0, 0), sp.Point(1, 0)))[0]
        tans.append(sp.Abs(line.slope))
        dists.append(sp.Abs(land.x - 0))
        # 배치 2: A(0,hA), B(1,hB) — 파편은 A→B 방향으로 진행
        line2 = sp.Line(sp.Point(0, hA), sp.Point(1, hB))
        land2 = line2.intersection(sp.Line(sp.Point(0, 0), sp.Point(1, 0)))[0]
        tans2.append(sp.Abs(line2.slope))
        dists2.append(sp.Abs(land2.x - 1))
        assert land2.x > 1  # 지면 도달점은 B 를 지나서
    s1, s2 = sp.nsimplify(sum(tans)), sp.nsimplify(sum(dists))
    a1, a2 = load_answer(1, "1"), load_answer(1, "2")
    # 공식 채점기준식 대조: tan θn = n^2+2n, a_n = 1/(n(n+2))
    f1 = all(sp.simplify(tans[k - 1] - (k * k + 2 * k)) == 0 for k in range(1, 9))
    f2 = all(sp.simplify(dists[k - 1] - sp.Rational(1, k * (k + 2))) == 0 for k in range(1, 9))
    ok = (s1 == a1 and s2 == a2 and sum(tans2) == s1 and sp.nsimplify(sum(dists2)) == s2 and f1 and f2)
    report(1, ok, f"(1) Σtan={s1} md={a1} / (2) Σa_n={s2} md={a2} (배치 뒤집어도 동일, 일반항 일치={f1 and f2})")


def p02():
    """일반 삼차 f=c3x^3+c2x^2+c1x+c0 에 f'(0)=f'(4)=0, f(0)=2, f(4)=1 을 풀고
    x=2 에서의 접선과 지면(y=0)의 교점 계산. A 에서 극대(하강 시작)·B 에서 극소인지,
    0<x<6 에서 접선이 지면 위에 있는지도 확인."""
    c = sp.symbols("c0:4")
    f = c[3] * x**3 + c[2] * x**2 + c[1] * x + c[0]
    sol = sp.solve([sp.diff(f, x).subs(x, 0), sp.diff(f, x).subs(x, 4), f.subs(x, 0) - 2, f.subs(x, 4) - 1], c, dict=True)
    assert len(sol) == 1
    F = f.subs(sol[0])
    d2 = sp.diff(F, x, 2)
    shape_ok = d2.subs(x, 0) < 0 and d2.subs(x, 4) > 0
    tan = F.subs(x, 2) + sp.diff(F, x).subs(x, 2) * (x - 2)
    roots = sp.solve(sp.Eq(tan, 0), x)
    hit = roots[0]
    ans = load_answer(2, "1")
    ok = len(roots) == 1 and hit == ans and shape_ok and hit > 2
    report(2, ok, f"f={sp.expand(F)}, 접선 y={sp.expand(tan)}, 충돌 x={hit}, md={ans}, A극대·B극소={shape_ok}")


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
