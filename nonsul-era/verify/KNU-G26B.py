#!/usr/bin/env python3
"""KNU 2026학년도 수시 논술(AAT) 자연계열Ⅱ 수학 (KNU-G26B-01~02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-G26B"
WIDTH = 2
RESULTS = []
x, h = sp.symbols("x h", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())


def load_answer(seq, i=0):
    return load_item(seq)["subparts"][i]["answer"]["value"]



def p01():
    msgs, ok = [], True
    A_, B_ = sp.symbols("A B", real=True)
    # 1-1: a 자연수 전수(1..200), b 는 (Ⅰ) g''(-2/3)=0 에서, (Ⅱ) g 가 증가(g'>=0, 판별식) + 변곡점 부호변화 확인
    sols = []
    for av in range(1, 201):
        bv = sp.solve(sp.diff(av * x**3 + B_ * x**2 + 2 * x - 15, x, 2).subs(x, sp.Rational(-2, 3)), B_)[0]
        g = av * x**3 + bv * x**2 + 2 * x - 15
        gp = sp.diff(g, x)
        mono = sp.minimum(gp, x, sp.S.Reals) >= 0
        g2 = sp.diff(g, x, 2)
        sign_change = g2.subs(x, -1) * g2.subs(x, 0) < 0
        if mono and sign_change:
            sols.append((av, bv))
    a11 = tuple(sp.sympify(load_answer(1, 0)))
    ok &= len(sols) == 1 and tuple(sols[0]) == a11
    msgs.append(f"1-1 (a,b)={sols} md={a11}")
    av, bv = sols[0]
    g = av * x**3 + bv * x**2 + 2 * x - 15
    # 1-2: r 범위 — (Ⅲ) s=g(2)-4-2r, (Ⅱ) f 증가 ⇔ h 꼭짓점 -r/2 <= 2 (g 는 증가, g(2-)=h(2)), (Ⅳ) 좌극한(두 인수 모두 x<2)
    r, c = sp.symbols("r c", real=True)
    t = sp.Symbol("t", negative=True)
    L = sp.limit((g.subs(x, 2 + 3 * t) - g.subs(x, 2 + t)) / t, t, 0, "-")
    hh = x**2 + r * x + (g.subs(x, 2) - 4 - 2 * r)
    c_of_r = sp.solve(sp.Eq(2 * sp.diff(hh, x).subs(x, c), L), c)[0]
    rng = sp.solve_univariate_inequality(c_of_r >= 2, r, relational=False).intersect(
        sp.solve_univariate_inequality(-r / 2 <= 2, r, relational=False))
    a12 = tuple(sp.sympify(load_answer(1, 1)))
    ok &= isinstance(rng, sp.Interval) and (rng.inf, rng.sup) == a12 and not rng.left_open and not rng.right_open
    msgs.append(f"1-2 r∈{rng} md={a12}")
    # 1-3(1)
    hb = hh.subs(r, rng.sup)
    a = lambda n: g.subs(x, n) if n < 2 else hb.subs(x, n)
    v31 = a(1) + a(5)
    a31 = sp.sympify(load_answer(1, 2))
    ok &= v31 == a31
    msgs.append(f"1-3(1) {v31} md={a31}")
    # 1-3(2): 정의식 (n+1)a_n = sum k b_{n-k+1} 을 n=1..80 순차로 풀어 b_n 을 직접 구함
    N = 80
    b = {}
    for n in range(1, N + 1):
        rest = sum(k * b[n - k + 1] for k in range(2, n + 1))
        b[n] = (n + 1) * a(n) - rest
    v32 = b[1] + 3 * b[2] + 4 * b[3] - b[50]
    a32 = sp.sympify(load_answer(1, 3))
    ok &= v32 == a32
    msgs.append(f"1-3(2) b1..b4={[b[i] for i in range(1,5)]} b50={b[50]} 값={v32} md={a32}")
    # 1-3(3): k>=4 에서 b_k 의 일차식(데이터로 보간·전 구간 확인) 후 극한을 sympy 로 직접 계산
    kk, n = sp.symbols("k n", positive=True, integer=True)
    lin = sp.interpolate([(4, b[4]), (5, b[5])], kk)
    assert all(lin.subs(kk, i) == b[i] for i in range(4, N + 1))
    head3 = sum(((b[i] - 44) / sp.Integer(12) + n / 2) ** 5 for i in (1, 2, 3))
    tail = sp.summation(sp.expand(((lin - 44) / 12 + n / 2) ** 5), (kk, 4, n + 3))
    lim = sp.limit((head3 + tail) / n**6, n, sp.oo)
    pq = sp.fraction(lim)
    a33 = tuple(sp.sympify(load_answer(1, 4)))
    ok &= lim.is_Rational and (pq[1], pq[0]) == a33
    msgs.append(f"1-3(3) b_k={lin}(k>=4) 극한={lim} (p,q)={(pq[1], pq[0])} md={a33}")
    report(1, ok, "; ".join(msgs))


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    msgs, ok = [], True
    # 2-1: U'(x)^2-1 = cos^m sin^n 을 만족하는 정수 (m,n) 를 -6..6 전수 탐색(기호 동치)
    U = sp.log((1 + sp.sin(x)) / sp.cos(x))
    lhs = sp.diff(U, x) ** 2 - 1
    hits = [(m, n) for m in range(-6, 7) for n in range(-6, 7)
            if all(abs(float((lhs - sp.cos(x) ** m * sp.sin(x) ** n).subs(x, v))) < 1e-12 for v in (0.3, 0.7, -0.4, 1.1))]
    a21 = tuple(sp.sympify(load_answer(2, 0)))
    ok &= hits == [a21]
    msgs.append(f"2-1 (m,n)={hits} md={a21}")
    # 함수 f, 길이 L, 접선 절편 c, 접선 위 거리 L 인 점 (a,b) — 정의에서 직접(수치)
    c6 = mp.cos(mp.pi / 6)
    f = lambda X: mp.log(mp.cos(X)) - 2 * mp.log(c6) if X < mp.pi / 6 else -mp.log(mp.cos(mp.pi / 3 - X))
    fp = lambda X: mp.diff(f, X)
    L = lambda T: mp.quad(lambda X: mp.sqrt(1 + fp(X) ** 2), [T, mp.pi / 6, mp.pi / 3] if T < mp.pi / 6 else [T, mp.pi / 3])
    def ab(T):
        d = fp(T); nrm = mp.sqrt(1 + d * d); l = L(T)
        return T + l / nrm, f(T) + l * d / nrm      # 접선 방향 (1,f') 쪽(a>=t), 거리 l
    cc = lambda T: f(T) - T * fp(T)
    # (1)
    v1 = mp.e ** L(mp.pi / 6)
    a221 = sp.sympify(load_answer(2, 1))
    ok &= abs(v1 - mp.mpf(sp.N(a221, 30))) < 1e-15
    msgs.append(f"2-2(1) e^L={mp.nstr(v1, 15)} md={a221}")
    # (2): 격자 600점 + 극값 후보 정밀화
    grid = [mp.pi / 3 * i / 600 for i in range(601)]
    vals = [mp.e ** cc(T) for T in grid]
    vmax = max(mp.e ** cc(mp.pi / 6), max(vals)); vmin = min(vals)
    a222 = sp.sympify(load_answer(2, 2))
    ok &= abs(vmax - mp.mpf(sp.N(a222[0], 30))) < 1e-12 and abs(vmin - mp.mpf(sp.N(a222[1], 30))) < 1e-12
    msgs.append(f"2-2(2) max={mp.nstr(vmax, 12)} min={mp.nstr(vmin, 12)} md=({sp.N(a222[0], 12)}, {sp.N(a222[1], 12)})")
    # (3): a'/b' 를 수치미분으로 6점에서 비교
    tt = sp.Symbol("t")
    e223 = sp.sympify(load_answer(2, 3), locals={"t": tt})
    err = 0
    for T in (mp.mpf("0.1"), mp.mpf("0.3"), mp.mpf("0.45"), mp.mpf("0.6"), mp.mpf("0.8"), mp.mpf("0.95")):
        da = mp.diff(lambda u: ab(u)[0], T); db = mp.diff(lambda u: ab(u)[1], T)
        err = max(err, abs(da / db - mp.mpf(sp.N(e223.subs(tt, sp.Float(str(T), 30)), 30))))
    ok &= err < 1e-10
    msgs.append(f"2-2(3) |a'/b' - md| 최대오차={mp.nstr(err, 3)}")
    # (4): Q 의 최대·최소 위치를 격자로 찾고 값 계산
    Qv = [ab(T)[0] ** 2 + ab(T)[1] ** 2 for T in grid]
    p_ = grid[Qv.index(max(Qv))]; q_ = grid[Qv.index(min(Qv))]
    q_ = mp.findroot(lambda T: mp.diff(lambda u: ab(u)[0] ** 2 + ab(u)[1] ** 2, T), q_ + mp.mpf("0.01")) if abs(q_ - mp.pi / 6) > 1e-2 else mp.pi / 6
    A_, B_ = ab(p_)
    v4 = mp.e ** A_ + mp.e ** B_ + q_
    a224 = sp.sympify(load_answer(2, 4))
    ok &= abs(v4 - mp.mpf(sp.N(a224, 30))) < 1e-10 and p_ == 0
    msgs.append(f"2-2(4) p={mp.nstr(p_, 6)} q={mp.nstr(q_, 10)} 값={mp.nstr(v4, 15)} md={a224}")
    report(2, ok, "; ".join(msgs))


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
