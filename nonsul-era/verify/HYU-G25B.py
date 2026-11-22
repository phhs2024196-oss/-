#!/usr/bin/env python3
"""HYU-G25B (한양대 2025학년도 논술 자연계열 오후1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from functools import lru_cache

import sympy as sp

PREFIX = "HYU-G25B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str) -> dict:
    """'키 = 식; 키 = 식' 형식 answer.value 를 사전으로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def turn_paths(N: int, turns: int) -> int:
    """(0,0)→(N,N) 최단 경로 중 방향 전환 횟수가 turns 인 경로 수(DP 전수)."""
    @lru_cache(maxsize=None)
    def go(i, j, d, k):            # d: 0=오른쪽, 1=위쪽 (마지막 이동 방향), k: 지금까지 전환 수
        if k > turns:
            return 0
        if i == N and j == N:
            return 1 if k == turns else 0
        s = 0
        if i < N:
            s += go(i + 1, j, 0, k + (d == 1))
        if j < N:
            s += go(i, j + 1, 1, k + (d == 0))
        return s
    return go(1, 0, 0, 0) + go(0, 1, 1, 0)


def p01():
    a, b, X0 = sp.symbols("a b x0", real=True)
    # (1) 5 이하 자연수 순서쌍 전수
    I = sp.integrate(2 * x**3 + 3 * a * x**2 + b * x, (x, 1, 2))
    cnt = sum(1 for av in range(1, 6) for bv in range(1, 6) if 20 <= I.subs({a: av, b: bv}) <= 30)
    ok1 = cnt == int(sp.sympify(md_answer(1, "1")))
    # (2) 격자 DP
    f6 = turn_paths(6, 4)
    g32 = turn_paths(32, 7)
    parts = md_parts(1, "2")
    ok2 = f6 == int(sp.sympify(parts["f(6)"])) and sp.sqrt(2 * g32) == sp.sympify(parts["sqrt(2*g(32))"])
    # 대표 오답: 한쪽 방향 시작만 센 값
    half = g32 // 2
    # (3) g(x)=c x (x-x0)(x-3), 조건으로 c, x0 결정 → f 조각별 정의 → ∫|f|
    c = sp.symbols("c", real=True)
    an = [None] + [9 * (1 - sp.Rational(2, 3)**(k - 1)) for k in range(1, 6)]
    gx = c * x * (x - X0) * (x - 3)

    def piece(k, G):               # [a_k, a_{k+1}) 위의 f, k=1..4
        if k == 1:
            return G
        prev = piece_cache[k - 1] if k - 1 in piece_cache else None
        # f(x) = (2/3)^{k-1} f((3/2)^{k-1}(x-a_{k+1})+3) — 인자는 [0,3) 에 들어가므로 f=g 사용
        return sp.Rational(2, 3)**(k - 1) * G.subs(x, sp.Rational(3, 2)**(k - 1) * (x - an[k + 1]) + 3)
    piece_cache = {}
    # x=a_2 에서 미분가능 조건 + g'(0)g'(3)=9, g'(0)>0
    f1 = gx
    f2 = piece(2, gx)
    dl = sp.diff(f1, x).subs(x, an[2])
    dr = sp.diff(f2, x).subs(x, an[2])
    sols = sp.solve([sp.Eq(dl, dr), sp.Eq(sp.diff(gx, x).subs(x, 0) * sp.diff(gx, x).subs(x, 3), 9)], [c, X0], dict=True)
    sols = [s for s in sols if sp.diff(gx, x).subs(x, 0).subs(s) > 0]
    assert len(sols) == 1, sols
    G = gx.subs(sols[0])
    fs = {k: sp.expand(piece(k, G)) for k in range(1, 5)}
    cont = all(sp.simplify(fs[k].subs(x, an[k + 1]) - fs[k + 1].subs(x, an[k + 1])) == 0 for k in range(1, 4))
    diff_ok = all(sp.simplify(sp.diff(fs[k], x).subs(x, an[k + 1]) - sp.diff(fs[k + 1], x).subs(x, an[k + 1])) == 0
                  for k in range(1, 4))
    total = 0
    for k in range(1, 5):
        rts = sorted({r for r in sp.solve(fs[k], x) if r.is_real and an[k] < r < an[k + 1]}, key=float)
        pts = [an[k]] + rts + [an[k + 1]]
        total += sum(sp.Abs(sp.integrate(fs[k], (x, pts[i], pts[i + 1]))) for i in range(len(pts) - 1))
    total = sp.nsimplify(sp.simplify(total))
    ok3 = cont and diff_ok and sp.simplify(total - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3,
           f"순서쌍 {cnt}, f(6)={f6}, g(32)={g32} (한쪽만 센 오답 {half}), g={sp.factor(G)}, 연속{cont}·미분{diff_ok}, ∫|f|={total}")


def p02():
    nn = sp.symbols("n", positive=True)
    # (1) cosθ = n/(n+1) 확인(수치) 후 두 접선의 x절편 차 × 높이
    for nv in (1, 3, 10):
        th = sp.nsolve(sp.sin(x) - sp.Rational(nv, nv + 1) * sp.tan(x), x, 1.0)
        assert abs(sp.cos(th) - sp.Rational(nv, nv + 1)) < 1e-12
    cth = nn / (nn + 1)
    sth = sp.sqrt(1 - cth**2)
    th = sp.Symbol("theta")
    m1 = cth                                  # (sin x)' = cos θ
    m2 = (nn / (nn + 1)) / cth**2             # (k tan x)' = k sec^2 θ
    x1 = th - sth / m1
    x2 = th - sth / m2
    S = sp.simplify(sp.Rational(1, 2) * sth * sp.Abs(x2 - x1))
    lim = sp.limit(nn**2 * S, nn, sp.oo)
    ok1 = sp.simplify(lim - sp.sympify(md_answer(2, "1"))) == 0
    # (2) 포물선 y^2=4px, P(x1, y1)
    p, X1 = sp.symbols("p x1", positive=True)
    res = []
    for sgn in (1, -1):
        Y1 = sgn * 2 * sp.sqrt(p * X1)
        Qy = 2 * p * (-p + X1) / Y1           # 접선 y1 y = 2p(x+x1) 과 x=-p
        Pp = sp.Matrix([Qy**2 / (4 * p), Qy])
        F = sp.Matrix([p, 0])
        u = sp.Matrix([X1, Y1]) - F
        v = Pp - F
        cosv = sp.simplify(u.dot(v) / sp.sqrt(u.dot(u) * v.dot(v)))
        res.append(sp.solve(sp.Eq(cosv, sp.Rational(1, 5)), X1))
    md2 = sp.sympify(md_answer(2, "2"), locals={"p": p})
    ok2 = all(len(r) == 1 and sp.simplify(r[0] - md2) == 0 for r in res)
    # (3) 정육면체 좌표: 아랫면 F(0,0,0) G(4,0,0) H(4,4,0) E(0,4,0), 윗면 A(0,4,4) B(0,0,4) C(4,0,4) D(4,4,4)
    s, q, l1, l2 = sp.symbols("s q l1 l2", real=True)
    A = sp.Matrix([0, 4, 4]); C = sp.Matrix([4, 0, 4]); D = sp.Matrix([4, 4, 4])
    P = sp.Matrix([s, s, 0]); Q = sp.Matrix([q, 0, 0])
    R1 = C + l1 * (P - C)
    R2 = D + l2 * (Q - D)
    eqs = list(R1 - R2) + [l2 - sp.Rational(4, 7)]   # DR:RQ = 4:3
    sol = [z for z in sp.solve(eqs, [s, q, l1, l2], dict=True)
           if 0 <= z[s] <= 4 and 0 <= z[q] <= 4 and 0 <= z[l1] <= 1]
    assert len(sol) == 1, sol
    Pv = P.subs(sol[0])
    nrm = (Pv - A).cross(C - A)
    cos3 = sp.Abs(nrm[2]) / nrm.norm()
    ok3 = sp.simplify(cos3 - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and ok2 and ok3, f"lim n^2 S_n={lim}, x1={res[0]}, P={list(Pv)}, cosθ={sp.nsimplify(cos3)}")


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
