#!/usr/bin/env python3
"""서울과기대 2025학년도 모의논술고사 문제 1~3 (SNUT-M25-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SNUT-M25"
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


import math
import numpy as np


def p01():
    # [1.1] a_n 을 직접 계산한 수치와 정적분 값 비교(리처드슨 외삽 없이 큰 n 과 정적분)
    def a_n(nn):
        return sum(math.log(1 + kk/nn) for kk in range(1, nn+1)) / nn   # ln(곱)=Σln
    def a_n_direct(nn):
        pr = sp.Integer(1)
        for kk in range(1, nn+1):
            pr *= (1 + sp.Rational(kk, nn))
        return sp.N(sp.log(pr)/nn, 20)
    chk_small = abs(a_n(7) - float(a_n_direct(7))) < 1e-12      # 곱→합 변환 확인
    I = sp.integrate(sp.log(1 + x), (x, 0, 1))
    v1, v2 = a_n(100000), a_n(200000)
    lim = 2*v2 - v1   # O(1/n) 오차 제거
    ans = load_answer(1, 0)
    ok1 = chk_small and same(I, ans) and abs(lim - float(sp.N(ans))) < 1e-9
    report(1, ok1, f"[1.1] ∫0^1 ln(1+x)dx={I}, a_n 외삽≈{lim:.10f}, yaml={ans}≈{float(sp.N(ans)):.10f}")
    # [1.2] f'(x) 부호, 부등식 무작위 검사
    f = sp.log(x)/x
    crit = sp.solve(sp.diff(f, x), x)
    inc_ok = sp.diff(f, x).subs(x, 1) > 0 and sp.diff(f, x).subs(x, 5) < 0 and crit == [sp.E]
    raw = load_answer(1, 1, raw=True)
    rel = raw.split(";")[1].strip()                        # a/b<log(a)/log(b)
    lhs, rhs = [sp.sympify(z, locals={"a": sp.Symbol("a"), "b": sp.Symbol("b")}) for z in rel.split("<")]
    import random
    random.seed(1)
    rel_ok = True
    for _ in range(2000):
        av = math.e + random.random()*50 + 1e-6; bv = av + random.random()*100 + 1e-6
        rel_ok &= float(lhs.subs({"a": av, "b": bv})) < float(rhs.subs({"a": av, "b": bv}))
    ok2 = inc_ok and rel_ok and ("0<x<e" in raw.replace(" ", "")) and ("x>e" in raw.replace(" ", ""))
    report(1, ok2, f"[1.2] 임계점={crit}, 증가/감소 부호 확인={inc_ok}, 무작위 2000쌍(e<a<b) a/b<ln a/ln b ={rel_ok}, yaml={raw}")
    # [1.3] 일반 등차수열로 극한 계산
    a1, b1, d1, d2, nn = sp.symbols("a1 b1 d1 d2 nn", positive=True)
    an = a1 + (nn - 1)*d1; bn = lambda m: b1 + (m - 1)*d2
    Sn = sp.summation(b1 + (k - 1)*d2, (k, 1, nn))
    expr = an**2 / ((bn(nn + 2) - bn(nn)) * Sn)
    L = sp.limit(expr, nn, sp.oo)
    r = sp.symbols("r", positive=True)                   # r = d1/d2
    sol = sp.solve(sp.Eq(sp.simplify(L.subs(d1, r*d2)), 27), r)
    ok3 = len(sol) == 1 and same(sol[0], load_answer(1, 2))
    report(1, ok3, f"[1.3] 극한={sp.simplify(L)}, d1/d2={sol}, yaml={load_answer(1,2)}")


def p02():
    f = lambda z: 5**z/(1 + 5**z) - sp.Rational(1, 2)
    # [2.1] [증명-수치확인] f(-x)+f(x)=0 : 격자 1000점 + 기호
    xs = np.linspace(-30, 30, 1000)
    ff = lambda z: 5.0**z/(1 + 5.0**z) - 0.5
    sym_ok = sp.simplify(f(-x) + f(x)) == 0
    num_ok = bool(np.all(np.abs(ff(xs) + ff(-xs)) < 1e-12))
    report(2, sym_ok and num_ok, f"[2.1] [증명-수치확인] f(-x)+f(x)=0 기호={sym_ok}, 격자 1000점={num_ok}")
    # [2.2] 교점 개수: 각 m 마다 h(x)=f(x)-mx 의 영점 개수(부호변화, 구간 [-L,L])
    def count_roots(m):
        L = 1/m + 5
        grid = np.linspace(-L, L, 400001)
        h = ff(grid) - m*grid
        s = np.sign(h)
        # 0 근처 처리: 정확히 x=0 은 h=0 → 별도 계산
        pos = grid[grid > 0]; hp = h[grid > 0]
        ch = int(np.sum(np.sign(hp[1:]) != np.sign(hp[:-1])))
        return 2*ch + 1
    thr = math.log(5)/4
    raw = load_answer(2, 1, raw=True)
    lo_s, hi_s = raw.replace(" ", "").split("<m<")
    lo = float(sp.sympify(lo_s)); hi = float(sp.sympify(hi_s, locals={"log": sp.log}))
    ms = list(np.linspace(0.005, 1.0, 120)) + [thr - 1e-3, thr + 1e-3, thr*0.999, thr*1.001, 0.2308]
    bad = [m for m in ms if (count_roots(m) == 3) != (lo < m < hi)]
    report(2, not bad and abs(hi - thr) < 1e-12 and lo == 0, f"[2.2] m 격자 125개에서 (교점 3개 ⇔ {lo_s}<m<{hi:.6f}) 불일치={bad}")
    # [2.3] 넓이 A(c)=∫_{-c}^{c}(f+1/2)dx = 2 → c; m=f(c)/c; 교점 a=-c 확인
    c = sp.symbols("c", positive=True)
    A = sp.integrate(5**x/(1 + 5**x), (x, -c, c))
    A_simpl = sp.simplify(A)
    csol = sp.solve(sp.Eq(A_simpl, 2), c)
    cv = csol[0]
    m_val = sp.simplify(f(cv)/cv)
    # 수치 검증: 직접 수치적분
    area_num = sp.N(sp.Integral(5**x/(1 + 5**x), (x, -cv, cv)), 20)
    mm = float(m_val)
    three = count_roots(mm) == 3 and mm < thr
    ok3 = same(m_val, load_answer(2, 2)) and abs(area_num - 2) < 1e-12 and three
    report(2, ok3, f"[2.3] A(c)={A_simpl}, c={cv}, m=f(c)/c={m_val}, 면적 수치={area_num}, 교점 3개={three}, yaml={load_answer(2,2)}")


def p03():
    th = sp.symbols("theta", positive=True)
    # [3.1] 좌표로 직접 계산: B(4/3,0), P=B+(cos(pi-th), sin(pi-th))?  A 는 B 의 왼쪽 → ∠ABP=θ 는 BA 방향(-x)과 BP 사이각
    tt = sp.symbols("tt", positive=True)
    P = sp.Matrix([sp.Rational(4, 3) - sp.cos(th), sp.sin(th)])   # BP 가 -x축과 θ 를 이룸 → 제1사분면
    A = sp.Matrix([sp.Rational(4, 3) - tt, 0]); O = sp.Matrix([0, 0])
    AP2 = sp.simplify((P - A).dot(P - A)); OP2 = sp.simplify(P.dot(P))
    raw = load_answer(3, 0, raw=True)
    e1, e2 = [sp.sympify(z.split("=")[1], locals={"t": tt, "theta": th}) for z in raw.split(";")]
    ok1 = sp.simplify(AP2 - e1) == 0 and sp.simplify(OP2 - e2) == 0
    report(3, ok1, f"[3.1] 좌표 계산 AP²={AP2}, OP²={OP2}, yaml={e1} ; {e2}")
    # [3.2] 모든 t∈(0,4/3) 에서 l1>l2 인 cosθ 범위: cosθ 격자 판정
    def holds(cv):
        ts = np.linspace(1e-9, 4/3 - 1e-9, 200001)
        # l1 = PA²+AB²+BP², l2 = OP²+7/9  (제곱합을 직접 식으로)
        l1 = (ts**2 + 1 - 2*ts*cv) + ts**2 + 1
        l2 = (25/9 - 8/3*cv) + 7/9
        return bool(np.all(l1 - l2 > 1e-13))
    lo_s, mid, hi_s = raw2 = load_answer(3, 1, raw=True).replace(" ", "").split("<")
    lo = float(sp.sympify(lo_s)); hi = float(sp.sympify(hi_s))
    cs = list(np.linspace(0.001, 0.999, 999)) + [2/3 - 1e-6, 2/3 + 1e-6]
    bad = [cv for cv in cs if holds(cv) != (lo < cv < hi)]
    report(3, not bad and lo == 2/3 and hi == 1 and mid == "cos(theta)", f"[3.2] cosθ 격자 {len(cs)}개 판정 불일치={bad[:3]}, 경계 2/3≈{lo:.6f}")
    # [3.3]
    cand = {"pi/6": sp.pi/6, "pi/4": sp.pi/4, "pi/3": sp.pi/3}
    good = sorted([kname for kname, v in cand.items() if sp.N(sp.cos(v)) > 2/3])
    ans = sorted(z.strip() for z in load_answer(3, 2, raw=True).split(","))
    report(3, good == ans, f"[3.3] cosθ>2/3 인 θ={good}, yaml={ans}")


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
