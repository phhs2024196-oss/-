#!/usr/bin/env python3
"""서울과기대 2026학년도 논술전형고사 문제 1~3 (SNUT-G26-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SNUT-G26"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)
SYM = {"x": x, "t": t, "n": n, "k": k, "pi": sp.pi, "e": sp.E}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def same(a, b, tol=1e-9):
    """기호 동일 또는 수치 근사 동일."""
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    return abs(complex(sp.N(a - b, 30))) < tol


import itertools
import math


def p01():
    import mpmath as mp
    mp.mp.dps = 40
    # [1.1] 원식 그대로 격자 평가 (x>0)
    def lhs(xv):
        L = mp.log(xv, 3)
        r1 = L*mp.log(81*xv, 3) + 4
        r2 = 4*L*mp.log(xv/3, 3) + 1
        return mp.sqrt(max(r1, 0)) + mp.sqrt(max(r2, 0)), min(r1, r2)
    raw = load_answer(1, 0, raw=True)
    lo_s, _, hi_s = raw.replace(" ", "").split("<=")[0], None, raw.replace(" ", "").split("<=")[2]
    lo = sp.sympify(lo_s); hi = sp.sympify(hi_s)
    bad = []; neg = 0
    grid = [mp.mpf(i)/200 for i in range(1, 2400)] + [mp.mpf(1)/3, mp.mpf(3), mp.mpf(1)/3 - mp.mpf(10)**-6, mp.mpf(3) + mp.mpf(10)**-6]
    for g in grid:
        v, mn = lhs(g)
        neg += (mn < -mp.mpf(10)**-30)
        inside = (v <= 4 + mp.mpf(10)**-25)
        if inside != (float(lo) - 1e-12 <= float(g) <= float(hi) + 1e-12):
            bad.append(float(g))
    ok1 = not bad and neg == 0
    report(1, ok1, f"[1.1] 격자 {len(grid)}점 불일치={bad[:3]}, 근호 안 음수 횟수={neg}, yaml 구간=[{lo},{hi}]")
    # [1.2]
    a_, b_ = sp.symbols("a_ b_", real=True)
    Xs, Ys = sp.symbols("Xs Ys", positive=True)
    sol = sp.solve([sp.Eq(Xs - Ys, 240), sp.Eq(Xs*Ys, 2**(2*(2 + 4)))], [Xs, Ys], dict=True)  # 2^(a+b)=64 → 4^a 4^b=4096
    sol = [s for s in sol if s[Xs] > 0 and s[Ys] > 0]
    av = sp.log(sol[0][Xs], 4); bv = sp.log(sol[0][Ys], 4)
    av, bv = sp.nsimplify(av), sp.nsimplify(bv)
    check = same(2**av*2**bv, 64) and same(4**av - 4**bv, 240)
    m = sp.symbols("m")
    mv = sp.solve(sp.Eq(bv, m*av + sp.Rational(2, 3)), m)[0]
    Ax = sp.solve(sp.Eq(0, mv*x + sp.Rational(2, 3)), x)[0]
    O = sp.Matrix([0, 0]); P = sp.Matrix([av, bv]); A = sp.Matrix([Ax, 0])
    u = O - P; w = A - P
    cosv = sp.simplify(u.dot(w)/(u.norm()*w.norm()))
    ok2 = check and same(cosv, load_answer(1, 1))
    report(1, ok2, f"[1.2] a={av}, b={bv}, m={mv}, A=({Ax},0), cos∠OPA={cosv}, yaml={load_answer(1,1)}")
    # [1.3]
    kk = sp.symbols("kk", integer=True, positive=True)
    def a_n(nn):
        return sp.integrate(sum(j*(j + 1)*x**(j - 1) for j in range(1, nn + 1)), (x, -1, 1))
    bn = [a_n(2*nn)/nn for nn in range(1, 26)]
    ok3 = same(sum(bn), load_answer(1, 2)) and all(same(a_n(2*nn), 2*nn*(nn + 1)) for nn in range(1, 10))
    report(1, ok3, f"[1.3] Σb_n(n=1..25) 직접 적분 합={sum(bn)}, a_2n=2n(n+1) 확인, yaml={load_answer(1,3-1)}")
    # [1.4] 열거
    players = set(range(1, 13))
    g1 = {4, 5, 6, 7, 8}
    g2s = [set(c) for c in itertools.combinations(sorted(players - g1), 5) if {1, 2, 3} <= set(c)]
    per_pair = set()
    union = set()
    for g2 in g2s:
        pool = sorted(players - g2 - {4})
        cnt = 0
        for g3 in itertools.combinations(pool, 5):
            cnt += 1
            union.add(g3)
        per_pair.add(cnt)
    ans = int(load_answer(1, 3))
    ok4 = (len(union) == ans)
    report(1, ok4, f"[1.4] 가능한 2경기 구성 {len(g2s)}가지, 쌍을 고정하면 3경기 라인업 {sorted(per_pair)}가지(전부 같음), 모든 가능한 쌍에 대한 라인업 합집합={len(union)}, yaml={ans} "
                   f"[공식해설 해석 주의: 28=합집합(쌍 미확정), 쌍 고정 시 6]")


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    # [2.1] sinα=sinβ, 0<α<β<π ⇒ β=π-α (α<π/2). 식을 α 에 대해 수치로 풀기
    h = lambda al: 4*mp.sin(al + 2*(mp.pi - al)) - 3*mp.cos(2*al + (mp.pi - al))
    roots = []
    N = 4000
    prev = None
    for i in range(1, N):
        al = mp.pi/2*i/N
        v = h(al)
        if prev is not None and prev[1]*v < 0:
            roots.append(mp.findroot(h, (prev[0], al), solver="bisect", tol=1e-25, maxsteps=200))
        prev = (al, v)
    kv = [mp.sin(r) for r in roots]
    ok1 = len(kv) == 1 and abs(kv[0] - mp.mpf(load_answer(2, 0).p)/load_answer(2, 0).q) < 1e-15
    report(2, ok1, f"[2.1] α 해={[mp.nstr(r,12) for r in roots]}, k=sinα={[mp.nstr(v,12) for v in kv]}, yaml={load_answer(2,0)}")
    kS = load_answer(2, 0)
    # [2.2]
    curve = sp.Rational(3, 8)*x**2 + 5*kS
    tx = sp.solve(sp.Eq(sp.diff(curve, x), -sp.Rational(3, 4)), x)[0]
    Py = curve.subs(x, tx)
    line = -sp.Rational(3, 4)*(x - tx) + Py
    Aq = sp.Point(sp.solve(line, x)[0], 0); Bq = sp.Point(0, line.subs(x, 0)); Oq = sp.Point(0, 0)
    tri = sp.Triangle(Oq, Aq, Bq)
    r = sp.simplify(tri.inradius)
    ok2 = same(r, load_answer(2, 1)) and same(tri.incenter.x, r) 
    report(2, ok2, f"[2.2] P=({tx},{Py}), l: y={sp.expand(line)}, A={Aq}, B={Bq}, 내접원 r={r}, yaml={load_answer(2,1)}")
    # [2.3] 평행이동한 내접원: 중심 (c, d), x축과 l 에 접, a=cx-r_c... 
    cx, cy = tri.incenter.x, tri.incenter.y
    aa, bb = sp.symbols("aa bb", real=True)
    ctr = (cx + aa, cy + bb)
    l_ab = sp.Line(Aq, Bq)
    lcoef = l_ab.coefficients  # (A,B,C)
    dist_l = sp.Abs(lcoef[0]*ctr[0] + lcoef[1]*ctr[1] + lcoef[2])/sp.sqrt(lcoef[0]**2 + lcoef[1]**2)
    sols = []
    for sy in (1, -1):
        eqs = [sp.Eq(ctr[1], sy*r)]
        bsol = sp.solve(eqs[0], bb)[0]
        for sgn in (1, -1):
            e2 = sp.Eq(sgn*(lcoef[0]*ctr[0] + lcoef[1]*ctr[1] + lcoef[2])/sp.sqrt(lcoef[0]**2 + lcoef[1]**2), r)
            asol = sp.solve(e2.subs(bb, bsol), aa)
            for av2 in asol:
                sols.append((sp.nsimplify(av2), sp.nsimplify(bsol)))
    # 직접 거리 재확인 + 조건 0<a<5, b<0 필터
    good = [(s_a, s_b) for (s_a, s_b) in sols if 0 < s_a < 5 and s_b < 0 and same(dist_l.subs({aa: s_a, bb: s_b}), r)]
    good = list(dict.fromkeys(good))
    raw = load_answer(2, 2, raw=True)
    ya, yb = [sp.sympify(z.split("=")[1]) for z in raw.split(";")]
    ok3 = len(good) == 1 and same(good[0][0], ya) and same(good[0][1], yb)
    report(2, ok3, f"[2.3] 후보 (a,b)={sols}, 조건(0<a<5,b<0) 만족={good}, yaml=({ya},{yb})")


def p03():
    t_ = sp.symbols("t", positive=True)
    al, be, a, b = sp.symbols("al be a b", real=True)
    f = lambda z: z**3 + a*z**2 + b*z
    eqs = [sp.diff(f(x), x).subs(x, al), sp.diff(f(x), x).subs(x, be),
           (2*al + be)/3, (f(be).subs(x, be) + 2*f(al).subs(x, al))/3 + t_**3] if False else None
    fx = x**3 + a*x**2 + b*x
    eqs = [sp.diff(fx, x).subs(x, al), sp.diff(fx, x).subs(x, be),
           (2*al + be)/3, (fx.subs(x, be) + 2*fx.subs(x, al))/3 + t_**3]
    sols = sp.solve(eqs, [al, be, a, b], dict=True)
    good = [s for s in sols if sp.simplify(s[al] - s[be]) != 0 and sp.simplify(s[be] - s[al]).subs(t_, 1) > 0 and
            sp.diff(fx.subs(s), x, 2).subs(x, s[al]).subs(t_, 1) < 0]
    raw = load_answer(3, 0, raw=True)
    ya, yb = [sp.sympify(z.split("=")[1], locals={"t": t_}) for z in raw.split(";")]
    ok1 = len(good) == 1 and sp.simplify(good[0][a] - ya) == 0 and sp.simplify(good[0][b] - yb) == 0
    report(3, ok1, f"[3.1] 연립해(극대 α<극소 β) 후보={len(sols)}, 채택={good}, yaml=({ya},{yb})")
    # [3.2] 구간 분할로 |f| 적분(수치 + 기호)
    fsol = fx.subs({a: ya, b: yb})
    roots = sp.solve(fsol, x)
    s_alpha, s_beta = good[0][al], good[0][be]
    parts = sorted(set(roots))
    S_sym = sp.integrate(sp.Abs(fsol.subs(t_, 1)), (x, s_alpha.subs(t_, 1), s_beta.subs(t_, 1)))
    S_expr = sp.integrate(fsol, (x, s_alpha, 0)) - sp.integrate(fsol, (x, 0, s_beta))
    ok2 = same(S_expr, load_answer(3, 1).subs(t, t_)) and same(S_sym, load_answer(3, 1).subs(t, 1))
    report(3, ok2, f"[3.2] S(t)={sp.simplify(S_expr)}, t=1 에서 |f| 직접 적분={S_sym}, yaml={load_answer(3,1)}")
    # [3.3]
    S = sp.simplify(S_expr)
    res = {}
    for kk in range(1, 8):
        L = sp.limit(S/t_**kk, t_, sp.oo)
        res[kk] = L
    kmin = min(kk for kk, L in res.items() if L.is_finite)
    raw3 = load_answer(3, 2, raw=True)
    yk, yv = [sp.sympify(z.split("=")[-1]) for z in raw3.split(";")]
    ok3 = (kmin == yk) and same(res[kmin], yv)
    report(3, ok3, f"[3.3] k=1..7 극한={ {k_: str(v) for k_, v in res.items()} }, 최소 k={kmin}, 극한={res[kmin]}, yaml=({yk},{yv})")


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
