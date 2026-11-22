#!/usr/bin/env python3
"""경북대 2024학년도 수시 논술(AAT) 자연계열Ⅱ (KNU-M24B-01~02) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp
import mpmath as mp

PREFIX = "KNU-M24B"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return None if v is None else sp.sympify(v)


def p01():
    import random
    random.seed(7)
    x0, x1, x2, X = sp.symbols("x0 x1 x2 X")
    F = sp.Function("F")
    b0, b1, b2, a0, a1 = sp.symbols("b0 b1 b2 a0 a1")
    # 1-1: 일반 보간으로 q(x0)-p(x0) / ((x0-x1)(x0-x2)) 를 b2 로 표현
    y0, y1, y2 = sp.symbols("y0 y1 y2")
    sq = sp.solve([b0 + b1 * x0 + b2 / 2 * x0 ** 2 - y0, b0 + b1 * x1 + b2 / 2 * x1 ** 2 - y1,
                   b0 + b1 * x2 + b2 / 2 * x2 ** 2 - y2], [b0, b1, b2], dict=True)[0]
    sp_ = sp.solve([a0 + a1 * x1 - y1, a0 + a1 * x2 - y2], [a0, a1], dict=True)[0]
    box = sp.simplify(((sq[b0] + sq[b1] * x0 + sq[b2] / 2 * x0 ** 2) - (sp_[a0] + sp_[a1] * x0)) / ((x0 - x1) * (x0 - x2)))
    ans = load_answer(1, 0).subs(sp.Symbol("b_2"), sq[b2])
    ok1 = sp.simplify(box - ans) == 0
    # 조건을 만족하는 시험 함수들: f'=1+s(1-cos)·k 등, f(0)=0, f'>=1, |f''|<=1
    tests = [lambda u: u + 0.4 * (u - mp.sin(u)), lambda u: 1.5 * u + 0.5 * (mp.sqrt(1 + u * u) - 1),
             lambda u: 1.5 * u + 0.5 * mp.log(mp.cosh(u))]
    d2 = lambda f, u: mp.diff(f, u, 2)
    d1 = lambda f, u: mp.diff(f, u)
    okc = all(d1(f, u) >= 1 - 1e-12 and abs(d2(f, u)) <= 1 + 1e-12 and abs(f(0)) < 1e-30
              for f in tests for u in [mp.mpf(k) / 4 for k in range(-20, 21)])
    # 1-2 [증명-수치확인]: f''(c)=b2 인 c 가 (min,max) 에 존재
    ok2 = True
    for f in tests:
        for _ in range(30):
            pts = sorted(random.uniform(-3, 3) for _ in range(3))
            if min(pts[1] - pts[0], pts[2] - pts[1]) < 0.05: continue
            xa, xb, xc = [mp.mpf(v) for v in pts]
            B2 = 2 * ((f(xc) - f(xb)) / (xc - xb) - (f(xb) - f(xa)) / (xb - xa)) / (xc - xa)
            grid = [xa + (xc - xa) * k / 400 for k in range(401)]
            vals = [d2(f, u) - B2 for u in grid]
            ok2 &= any(vals[i] == 0 or vals[i] * vals[i + 1] <= 0 for i in range(400))
    # 1-3 [증명-수치확인]: 할선법, (ㄷ) 만족하는 시작값에서 |t6|<=1/128
    ok3 = True; worst = 0; ntest = 0
    for f in tests:
        for _ in range(200):
            t1 = random.uniform(0.05, 0.999); t2 = random.uniform(0.01, t1)
            T = [None, mp.mpf(t1), mp.mpf(t2)]
            for n in range(1, 5):
                a_, b_ = T[n], T[n + 1]
                T.append(b_ if a_ == b_ else b_ - f(b_) * (b_ - a_) / (f(b_) - f(a_)))
            if not (1 > T[1] > T[2] > T[3] > T[4] > 0): continue
            ntest += 1
            worst = max(worst, abs(T[6]) * 128)
            ok3 &= abs(T[6]) <= mp.mpf(1) / 128
    report(1, ok1 and okc and ok2 and ok3 and ntest > 50,
           f"□-b2/2 일치={ok1}, 시험함수 조건={okc}, 1-2 c 존재(무작위 90회)={ok2}, 1-3 (ㄷ) 만족 {ntest}건 max 128|t6|={mp.nstr(worst,4)} [증명-수치확인]")


def p02():
    l, m, n = sp.symbols("l m n")
    a, b, c = sp.symbols("a b c")
    # 2-1: 기울기 정의로 연립
    sol = sp.solve([(a**2 - c**2) / (a - c) - l, (b**2 - c**2) / (b - c) - m, (a**2 - b**2) / (a - b) - n], [a, b, c], dict=True)[0]
    A1 = load_answer(2, 0)
    ok1 = all(sp.simplify(sol[v] - A1[i]) == 0 for i, v in enumerate((a, b, c)))
    # 2-2: 직접 적분
    x1, x2 = sp.symbols("x1 x2")
    seg = lambda u, w: sp.integrate((u + w) * x - u * w - x**2, (x, u, w))
    D = seg(sol[b], sol[a]) + seg(sol[a], sol[c])
    ok2 = sp.simplify(D - load_answer(2, 1)) == 0
    # 2-3(1),(2),(3) 독립 수치: A 의 x좌표 a 를 주고 b<a<c 정삼각형을 수치로 찾음
    mp.mp.dps = 30
    def tri(av):
        av = mp.mpf(av)
        def eqs(bv, cv):
            P = lambda u: (u, u * u)
            d2 = lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            return [d2(P(av), P(bv)) - d2(P(bv), P(cv)), d2(P(av), P(cv)) - d2(P(bv), P(cv))]
        best = None
        span = 2 * abs(av) + 3
        for bg in [av - span * k / 30 for k in range(1, 31)]:
            for cg in [av + span * k / 30 for k in range(1, 31)]:
                try:
                    r = mp.findroot(eqs, (bg, cg))
                except Exception:
                    continue
                if r[0] < av < r[1] and max(abs(v) for v in eqs(r[0], r[1])) < 1e-20:
                    best = (r[0], r[1]); break
            if best: break
        return best
    ms, ok3 = [], True
    for av in [-20, -5, -2, -1, -0.3, 0, 0.4, 1.5, 3, 10, 40]:
        bc = tri(av)
        ok3 &= bc is not None
        if not bc: continue
        mv = bc[0] + bc[1]
        side = mp.sqrt((bc[1] - bc[0]) ** 2 + (bc[1] ** 2 - bc[0] ** 2) ** 2)
        ms.append((av, mv, side))
    al, be = [float(v) for v in load_answer(2, 2)]
    ok3 &= all(al < float(v[1]) < be for v in ms)
    ok3 &= all(ms[i][1] < ms[i + 1][1] for i in range(len(ms) - 1))       # a 증가 ⇔ m 증가(단사)
    ok3 &= float(ms[0][1]) < al + 0.05 and float(ms[-1][1]) > be - 0.05    # 끝에서 α,β 에 접근(전사)
    # 2-3(3): 1/R 을 수치 삼각형에서 직접 얻어 적분 — m→a 역함수 대신 구간 적분을 직접 수행
    Rm = lambda mv: 2 * mp.sqrt(3) * (1 + mv * mv) ** 1.5 / (1 - 3 * mv * mv)
    okR = all(abs(Rm(v[1]) - v[2]) < 1e-15 for v in ms)       # 변 길이 공식 검증(독립 수치 삼각형과 비교)
    I = mp.quad(lambda mv: 1 / Rm(mv), [al, be])
    ok4 = okR and abs(I - mp.mpf(sp.N(load_answer(2, 4), 40))) < 1e-20
    report(2, ok1 and ok2 and ok3 and ok4,
           f"2-1 일치={ok1}, D1+D2 일치={ok2}, 수치 정삼각형 {len(ms)}개 m 범위·단조={ok3}, R(m) 공식={okR}, ∫={mp.nstr(I,15)}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda q: int(q[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
