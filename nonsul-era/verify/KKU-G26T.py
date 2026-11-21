#!/usr/bin/env python3
"""KKU-G26T (건국대 2026학년도 KU논술우수자 통합(수학)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KKU-G26T"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _parts(s: str) -> dict:
    out = {}
    for seg in s.split(";"):
        k, v = seg.split("=", 1) if "=" in seg and not seg.strip().startswith("(") else (None, seg)
        out[(k or "").strip()] = v.strip()
    return out


def p02():
    msgs, oks = [], []
    # ── 2-1: 원 T 중심 C(0,c), A(t,t^2), 넓이 1/4, A 에서 공통접선 ──
    t, c, x, y = sp.symbols("t c x y", real=True)
    tp = sp.symbols("tp", positive=True)
    sols = []
    for cc in sp.solve(sp.Eq(sp.Rational(1, 2) * 2 * tp * sp.Abs(c - tp**2), sp.Rational(1, 4)), c):
        # 공통접선 ⇔ CA ⊥ 접선(기울기 2t): (t^2-c)/t * 2t = -1
        for tv in sp.solve(sp.Eq((tp**2 - cc) / tp * 2 * tp, -1), tp):
            sols.append((tv, cc.subs(tp, tv)))
    assert len(sols) == 1, sols
    tv, cv = sols[0]
    A = (tv, tv**2)
    r2 = A[0]**2 + (A[1] - cv)**2
    # 넓이: 직접 적분(원의 아래쪽 호 − 포물선), 대칭 2배
    lower = cv - sp.sqrt(r2 - x**2)
    S = sp.simplify(2 * sp.integrate(lower - x**2, (x, 0, A[0])))
    pa = _parts(md_answer(2, "1"))
    mdA = sp.sympify(pa["A"])
    mdS = sp.sympify(pa["S"])
    ok1 = sp.simplify(mdA[0] - A[0]) == 0 and sp.simplify(mdA[1] - A[1]) == 0 and sp.simplify(S - mdS) == 0
    # 공식 해설 경로(거리=반지름)도 같은 t 인지
    cgen = tp**2 + 1 / (4 * tp)
    dist = (cgen + tp**2) / sp.sqrt(1 + 4 * tp**2)                    # C 에서 y=2tx-t^2 까지 거리
    tdist = sp.solve(sp.Eq(dist**2, tp**2 + 1 / (16 * tp**2)), tp)
    ok1 = ok1 and tdist == [tv]
    oks.append(ok1); msgs.append(f"2-1 A={A} C=(0,{cv}) S={S}")
    # ── 2-2: 전수 열거 ──
    def pq(n):
        Y = range(1, n + 1)
        p = sum(1 for f in itertools.product(Y, repeat=3) if all(f[a - 1] > 2 * a for a in (1, 2, 3)))
        q = sum(1 for f in itertools.product(Y, repeat=3) if f[0] < f[1] < f[2])
        return p, q
    p7, q7 = pq(7)
    nmin = next(n for n in range(7, 60) if (lambda u: u[0] - u[1])(pq(n)) > 0)
    n = sp.symbols("n")
    poly_ok = all(sp.expand((n - 2) * (n - 4) * (n - 6) - n * (n - 1) * (n - 2) / 6 - (lambda u: u[0] - u[1])(pq(k))).subs(n, k) == 0
                  for k in range(7, 15))
    m = re.findall(r"-?\d+", md_answer(2, "2").replace("(1)", "").replace("(2)", ""))
    ok2 = [int(v) for v in m] == [p7 - q7, nmin] and poly_ok
    oks.append(ok2); msgs.append(f"2-2 p7={p7} q7={q7} p-q={p7-q7} n_min={nmin} p8-q8={pq(8)[0]-pq(8)[1]}")
    # ── 2-3: f = x^3 + ux^2 + vx (최고차 1, f(0)=0), f(3)=12, argmax 넓이 = 4/3 ──
    u, v, s = sp.symbols("u v s", real=True)
    f = x**3 + u * x**2 + v * x
    eqs = [sp.Eq(f.subs(x, 3), 12), sp.Eq(sp.diff(f - 4 * x, x).subs(x, sp.Rational(4, 3)), 0)]
    sol = sp.solve(eqs, [u, v], dict=True)[0]
    fs = f.subs(sol)
    # 넓이 = (1/2)|OA| * dist(P, y=4x) ∝ |f(t)-4t| ; (0,3) 에서 최대점 확인
    h = sp.lambdify(x, fs - 4 * x)
    grid = [3 * k / 30000 for k in range(1, 30000)]
    tmax = max(grid, key=lambda z: abs(h(z)))
    roots = sorted(r for r in sp.solve(fs, x) if r != 0)
    md3 = sorted(sp.sympify(md_answer(2, "3").strip("{} ").split(",")[i]) for i in range(2))
    ok3 = roots == md3 and abs(tmax - 4 / 3) < 1e-3 and all(r > 3 for r in roots) and len(set(sp.solve(fs, x))) == 3
    oks.append(ok3); msgs.append(f"2-3 f={sp.expand(fs)} roots={roots} argmax~{tmax:.4f}")
    report(2, all(oks), " | ".join(msgs))


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
