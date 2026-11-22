#!/usr/bin/env python3
"""EWHA-M27B (이화여대 2027학년도 모의논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import random
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "EWHA-M27B"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    mp.mp.dps = 30
    f = lambda x: abs(x) * mp.e ** abs(x)
    # (1) x>0 에서 f'' 기호 계산, (2) [증명-수치확인] 현이 곡선 위
    x = sp.symbols("x", positive=True)
    f2 = sp.simplify(sp.diff(x * sp.exp(x), x, 2))
    ok1 = sp.simplify(f2 - (x + 2) * sp.exp(x)) == 0
    random.seed(2)
    ok2 = True
    for _ in range(3000):
        x1, x2 = sorted(random.uniform(-4, 4) for _ in range(2))
        if x2 - x1 < 1e-3:
            continue
        for t in (0.1, 0.37, 0.5, 0.81):
            xm = x1 + t * (x2 - x1)
            if not f(xm) < (1 - t) * f(x1) + t * f(x2):
                ok2 = False
    # S(a) 수치 정의
    def S(a):
        a = mp.mpf(a)
        pts = [a, 0, a + 1] if a < 0 < a + 1 else [a, a + 1]
        return (f(a) + f(a + 1)) / 2 - mp.quad(f, pts)
    dS = lambda a, h, s: (S(a + s * h) - S(a)) / (s * h)
    # (3) 좌우 미분계수가 다른 점: 격자 위 점들의 좌·우 차분 비교
    kinks = []
    for a in [i / 4 for i in range(-12, 9)]:
        r, l = dS(a, mp.mpf("1e-8"), 1), dS(a, mp.mpf("1e-8"), -1)
        if abs(r - l) > 1e-3:
            kinks.append(a)
    ans3 = sorted(float(v) for v in sp.sympify(md_answer(1, "3").replace("{", "[").replace("}", "]")))
    ok3 = kinks == ans3 and abs(dS(0, mp.mpf("1e-10"), 1) - 0.5) < 1e-6
    # (4) 최솟값: 격자 + 국소 정밀화
    grid = [i / 200 for i in range(-800, 601)]
    vals = [S(a) for a in grid]
    smin = min(vals)
    ok4 = abs(smin - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "4")), 30)))) < 1e-12
    # (5) S(a)=c 근 개수 = 3 이 되는 c: 극댓값 (−1,0) 구간 최대 → 그 값에서 근 3개, 주변은 4/2개
    amax = mp.findroot(lambda a: mp.diff(S, a), -0.4)
    cstar = S(amax)
    c_md = mp.mpf(str(sp.N(sp.sympify(md_answer(1, "5")), 30)))

    def nroots(c):
        g = [S(a) - c for a in grid]
        n = sum(1 for u, w in zip(g, g[1:]) if u * w < 0)
        n += sum(1 for u in g if abs(u) < 1e-20)
        return n
    ok5 = abs(cstar - c_md) < 1e-12 and nroots(cstar - mp.mpf("1e-6")) == 4 and nroots(cstar + mp.mpf("1e-6")) == 2 \
        and abs(amax + 0.5) < 1e-10
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] kinks={kinks} min={mp.nstr(smin, 12)} c*={mp.nstr(cstar, 12)} at a={mp.nstr(amax, 8)}")


def p02():
    k = sp.symbols("k", integer=True, positive=True)
    exprs = [sp.sympify(md_answer(2, s), locals={"k": k}) for s in ("1", "2", "3")]
    ok = True
    for kv in (2, 3, 4, 5):
        n = 2 * kv
        cnt = {}
        for p in itertools.permutations(range(1, n)):           # 0 을 고정해 회전 동일 제거
            seq = (0,) + p
            pos = [0] * n
            for i, v in enumerate(seq):
                pos[v] = i
            adj = sum(1 for c in range(kv) if (pos[2 * c] - pos[2 * c + 1]) % n in (1, n - 1))
            cnt[adj] = cnt.get(adj, 0) + 1
        got = [cnt.get(kv, 0), cnt.get(kv - 1, 0), cnt.get(kv - 2, 0)]
        want = [int(e.subs(k, kv)) for e in exprs]
        ok = ok and got == want
    # (4) 정확한 유리수로 부분합 계산
    a = lambda kv: int(exprs[1].subs(k, kv))
    good = [nv for nv in range(2, 40)
            if 0 < sum(sp.Rational(a(kv), 2**(kv - 1) * 3**(kv + 1)) for kv in range(2, nv + 1)) < 1]
    want4 = sorted(int(v) for v in sp.sympify(md_answer(2, "4").replace("{", "[").replace("}", "]")))
    # 단조 증가(합의 일반항 ≥ 0)로 n≥40 도 1 초과 유지
    ok4 = good == want4 and all(a(kv) >= 0 for kv in range(2, 60))
    report(2, ok and ok4, f"전수(k=2..5) 일치, n={good}")


def p03():
    c, al = sp.symbols("c alpha", positive=True)
    X, Y = sp.symbols("X Y", real=True)
    kk = sp.symbols("k", positive=True, integer=True)
    ok = True
    hk_md = sp.sympify(md_answer(3, "2"), locals={"k": kk})
    for kv in range(1, 7):
        xc = c / Y**kv                                    # x = c / y^k
        slope = sp.diff(xc, Y).subs(Y, al)
        tang = (c / al**kv) + slope * (Y - al)            # x = ... 접선
        a = sp.simplify(tang.subs(Y, 0))                  # x 절편
        b = sp.solve(sp.Eq(tang, 0), Y)[0]                # y 절편
        rh = 2 * a * b
        rect = 4 * (c / al**kv) * al
        h = sp.simplify(rh / rect)
        ok = ok and h.free_symbols == set() and sp.simplify(h - hk_md.subs(kk, kv)) == 0
        if kv == 1:
            area = sp.simplify(rh)
            parts = dict(s.split("=", 1) for s in md_answer(3, "1").replace(" ", "").split(";"))
            ok = ok and sp.simplify(area - sp.sympify(parts["area"], locals={"c": c})) == 0
            # ab=4c 에서 둘레 최소
            t = sp.symbols("t", positive=True)
            P = 4 * sp.sqrt(t**2 + (4 * c / t)**2)
            cr = sp.solve(sp.diff(P, t), t)
            pmin = sp.simplify(P.subs(t, cr[0]))
            ok = ok and sp.simplify(pmin - sp.sympify(parts["min"], locals={"c": c})) == 0
            # 접함 조건 수치 확인: c=1, ab<4 → 교점 0, ab>4 → 8
            xs = sp.symbols("xs", positive=True)
            n_int = lambda av, bv: 4 * len(sp.solve(sp.Eq(xs / av + 1 / (xs * bv), 1), xs))
            ok = ok and n_int(2, 2) == 4 and n_int(1, 3) == 0 and n_int(2, 3) == 8
    n = sp.symbols("n", positive=True, integer=True)
    term = 4 / (kk**2 * hk_md.subs(kk, kk + 1))
    ssum = sp.limit(sp.summation(sp.apart(sp.simplify(term), kk), (kk, 1, n)), n, sp.oo)
    ok3 = sp.simplify(ssum - sp.sympify(md_answer(3, "3"))) == 0
    report(3, ok and ok3, f"h_k={hk_md} (k=1..6 기호 일치) sum={ssum}")


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
