#!/usr/bin/env python3
"""KNU-G21B (경북대 2021 수시 논술 자연계열Ⅱ 수학 1·2·3) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KNU-G21B"
WIDTH = 2
RESULTS = []
mp.mp.dps = 40


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    """소문항 sub(0부터)의 answer.value 를 ';' 로 나눠 sympy 식 목록으로."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return [sp.sympify(s.strip()) for s in str(v).split(";")]




X, K = sp.symbols("x k", real=True)
F1 = (X - sp.Rational(2, 3)) * (X - 2) ** 3
F2 = (X + sp.Rational(2, 3)) * (X - 2) ** 3 + (2 - K) * (X - 2) ** 2


def _g(t, kv):
    """방정식 f(x)=t (0<=x<=3) 의 서로 다른 실근 개수(유리수 t, k 에서 정확히)"""
    r1 = [r for r in sp.Poly(sp.expand(F1 - t), X).real_roots() if 0 <= r < 2]
    r2 = [r for r in sp.Poly(sp.expand(F2.subs(K, kv) - t), X).real_roots() if 2 <= r <= 3]
    return len(set(r1) | set(r2))


def p01():
    oks, msgs = [], []
    h = sp.Symbol("h")
    L = sp.limit((F1.subs(X, 2 + h) - 0) / h, h, 0, "-"); R = sp.limit((F2.subs(X, 2 + h) - 0) / h, h, 0, "+")
    (a1,) = load_answer(1, 0)
    oks.append(F1.subs(X, 2) == 0 == F2.subs(X, 2) and L == R == a1); msgs.append(f"1-1 좌={L} 우={R}")
    # 1-2: k 격자에서 극대 존재(f' 부호 + → -) 여부
    def hasmax(kv):
        xs = [mp.mpf(i) / 200 for i in range(-400, 1400)]
        fp = lambda x: float(sp.diff(F1, X).subs(X, x)) if x < 2 else float(sp.diff(F2, X).subs({X: x, K: kv}))
        s = [fp(x) for x in xs]
        return any(s[i] > 0 and s[i + 1] <= 0 and (i + 2 >= len(s) or s[i + 2] < 0) for i in range(len(s) - 1))
    ks = [sp.Rational(i, 4) for i in range(-4, 21)]
    hm = [kv for kv in ks if hasmax(kv)]
    (a2,) = load_answer(1, 1)
    oks.append(hm == [kv for kv in ks if kv > a2]); msgs.append(f"1-2 극대 존재 k 격자 최소={min(hm)}")
    # 1-3: k 격자에서 두 조건 직접 판정
    eps = sp.Rational(1, 10 ** 6)
    good = []
    for kv in [sp.Rational(i, 4) for i in range(0, 49)]:
        t0 = sp.Rational(-1, 3)
        c1 = _g(t0 + eps, kv) - _g(t0 - eps, kv) == _g(t0, kv)
        # 최솟값 c: 임계점·끝점 후보
        fp2 = sp.diff(F2.subs(K, kv), X)
        cand = [sp.Integer(0), sp.Integer(1), sp.Integer(2), sp.Integer(3)] + [r for r in sp.Poly(fp2, X).real_roots() if 2 <= r <= 3]
        val = lambda x: (F1 if x < 2 else F2.subs(K, kv)).subs(X, x)
        vals = [sp.N(val(x), 50) for x in cand]
        c = min(vals)
        gc = len({sp.N(x, 30) for x, v in zip(cand, vals) if abs(v - c) < 1e-30})
        cr = sp.nsimplify(c, rational=True, tolerance=1e-40)
        c2 = _g(cr + eps, kv) - _g(cr - eps, kv) > gc
        if c1 and c2:
            good.append(kv)
    b, gm = load_answer(1, 2)
    oks.append(good == [kv for kv in [sp.Rational(i, 4) for i in range(0, 49)] if b < kv < gm]); msgs.append(f"1-3 만족 k 격자=[{min(good)},{max(good)}] 개수 {len(good)}")
    report(1, all(oks), "; ".join(msgs))


def _bis(F, lo, hi, n=400):
    """F(lo), F(hi) 부호가 다를 때 이분법"""
    flo = F(lo)
    for _ in range(n):
        mid = (lo + hi) / 2
        if (F(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _a2(t):
    # y=ln x 로 놓고 y - t e^y = 0 의 큰 근(y > ln(1/t))
    return mp.e ** _bis(lambda y: y - t * mp.e ** y, mp.log(1 / t), mp.mpf(400))


def _geo(t):
    t = mp.mpf(t)
    f = lambda x: mp.log(x) - t * x
    a1 = _bis(f, mp.mpf(1), 1 / t)
    a2 = _a2(t)
    T1 = lambda x: (x - a1) / a1 + mp.log(a1); T2 = lambda x: (x - a2) / a2 + mp.log(a2)
    xr = _bis(lambda x: T1(x) - T2(x), a1, a2); R = mp.matrix([xr, T1(xr)])
    u = mp.matrix([a1, mp.log(a1)]) - R; w = mp.matrix([a2, mp.log(a2)]) - R
    th = mp.acos((u.T * w)[0] / (mp.norm(u) * mp.norm(w)))
    S = mp.quad(lambda x: min(T1(x), T2(x)) - mp.log(x), [a1, xr, a2])
    return a1, a2, xr, abs(mp.tan(th)), S


def p02():
    oks, msgs = [], []
    mp.mp.dps = 50
    # 2-1: 교점 2개인 t 의 상한(이분법; 교점 수는 ln x - tx 의 최댓값 부호로가 아니라 부호변화 격자로 셈)
    def two(t):
        xs = [mp.mpf(1) / 1000 * mp.mpf("1.002") ** i for i in range(20000)]
        v = [mp.log(x) - t * x for x in xs]
        return sum(1 for i in range(len(v) - 1) if v[i] * v[i + 1] < 0) == 2
    lo, hi = mp.mpf("0.1"), mp.mpf("0.5")
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if two(mid) else (lo, mid)
    (b,) = load_answer(2, 0)
    oks.append(abs(lo - sp.N(b, 30)) < 1e-5); msgs.append(f"2-1 β≈{mp.nstr(lo,10)}")
    # 2-2: 두 t 에서 기하로 구한 |tanθ|, S 로 k,l / m,n 을 연립
    rows1, rows2 = [], []
    for t in ["0.1", "0.25", "0.33"]:
        a1, a2, xr, tn, S = _geo(t)
        rows1.append((a1, a2, tn)); rows2.append((a1, a2, mp.mpf(t), S))
    kk, ll = mp.lu_solve(mp.matrix([[r[2], r[2] * r[0] * r[1]] for r in rows1[:2]]), mp.matrix([r[1] - r[0] for r in rows1[:2]]))
    ok_ext = abs(rows1[2][2] * (kk + ll * rows1[2][0] * rows1[2][1]) - (rows1[2][1] - rows1[2][0])) < 1e-20
    m_, n_ = mp.lu_solve(mp.matrix([[(r[1] - r[0]) / 2, (r[1] - r[0]) / 2 * r[2] ** 2 * r[0] * r[1]] for r in rows2[:2]]), mp.matrix([r[3] for r in rows2[:2]]))
    r = rows2[2]; ok_ext &= abs((r[1] - r[0]) / 2 * (m_ + n_ * r[2] ** 2 * r[0] * r[1]) - r[3]) < 1e-15
    A, B = load_answer(2, 1)
    oks.append(ok_ext and abs(kk + ll - float(A)) < 1e-15 and abs(m_ * n_ - float(B)) < 1e-12)
    msgs.append(f"2-2 k={mp.nstr(kk,8)} l={mp.nstr(ll,8)} m={mp.nstr(m_,8)} n={mp.nstr(n_,8)}")
    # 2-3(1) [증명-수치확인]: t 격자에서 t^-1<a2<t^-5/3
    ok3 = True
    for t in [mp.mpf(10) ** (-j) for j in range(1, 20)] + [mp.mpf("0.3"), mp.mpf("0.36")]:
        a2 = _a2(t)
        ok3 &= 1 / t < a2 < t ** (-mp.mpf(5) / 3)
    oks.append(ok3); msgs.append(f"2-3(1) [증명-수치확인]={ok3}")
    # 2-3(2): 기하로 직접 비율 계산, t→0+
    vals = []
    for j in [4, 8, 12]:
        t = mp.mpf(10) ** (-j)
        a1, a2, xr, tn, S = _geo(t)
        vals.append(xr * tn / (t * S))
    (L,) = load_answer(2, 2)
    oks.append(abs(vals[-1] - float(L)) < 1e-6 and abs(vals[-1] - L) < abs(vals[0] - L)); msgs.append(f"2-3(2) 비율 t=1e-4,1e-8,1e-12: {[mp.nstr(v,10) for v in vals]}")
    report(2, all(oks), "; ".join(msgs))


def p03():
    oks, msgs = [], []
    mp.mp.dps = 30
    import random
    random.seed(3)
    hfun = lambda x: 1 / (1 + (1 + x) ** 2)
    funcs = [mp.exp, lambda x: x ** 2, hfun, lambda x: (x - 1.3) ** 4]
    # 3-1 [증명-수치확인]
    ok1 = True
    for f in funcs:
        for _ in range(15):
            a = random.uniform(0, 3); b = a + random.uniform(0.1, 3); x = random.uniform(a, b)
            ok1 &= f(x) <= (x - a) / (b - a) * f(b) + (b - x) / (b - a) * f(a) + 1e-20
            L = random.uniform(0.1, 2); t = L + random.uniform(0, 2)
            I = mp.quad(f, [t - L, t + L]) / (2 * L)
            ok1 &= f(t) <= I + 1e-20 and I <= (f(t - L) + f(t + L) + 2 * f(t)) / 4 + 1e-20
    oks.append(ok1); msgs.append(f"3-1 [증명-수치확인]={ok1}")
    # 3-2(1): p(0),p(1), (ㄷ) 결정과 q(s) 인수분해 확인
    a, b, s_ = sp.symbols("a b s", nonnegative=True); c = sp.Symbol("c")
    h = lambda x: 1 / (1 + (1 + x) ** 2)
    p = s_ * h(a) + (1 - s_) * h(b) - h(s_ * a + (1 - s_) * b)
    A, B = 1 + a, 1 + b
    csol = sp.solve(sp.Poly(sp.expand(s_ * A + (1 + c * s_) * B - (1 + s_ * a + (1 - s_) * b)), s_, a, b).coeffs(), c)
    cval = csol[c] if isinstance(csol, dict) else csol[0]
    q = sp.factor(sp.simplify(p * (1 + A ** 2) * (1 + B ** 2) * (1 + (s_ * A + (1 - s_) * B) ** 2)))
    qq = (B - A) ** 2 * s_ * (1 - s_) * (s_ * A ** 2 + (1 - s_) * B ** 2 + 2 * A * B - 1)
    okq = sp.expand(q - qq) == 0
    g1, g2, g3, ell = load_answer(3, 1)
    oks.append([sp.simplify(p.subs(s_, 0)), sp.simplify(p.subs(s_, 1)), cval] == [g1, g2, g3] and okq)
    msgs.append(f"3-2(1) p(0)={sp.simplify(p.subs(s_,0))} p(1)={sp.simplify(p.subs(s_,1))} (ㄷ)={cval} q 인수분해 일치={okq}")
    x = sp.Symbol("x")
    I = sp.integrate(h(x), (x, 0, 2))
    oks.append(sp.floor(25 * I) == ell and sp.diff(h(x), x, 2).subs(x, 0) > 0); msgs.append(f"3-2(2) ∫={sp.simplify(I)}≈{sp.N(I,10)} 25∫={sp.N(25*I,8)}")
    # 3-3 [증명-수치확인]: c_k 단조 감소
    ok3 = True
    for f in funcs:
        L = 1.7; t = 1.2
        ck = [mp.mpf(2) ** (k - 1) / L * mp.quad(f, [t - L / 2 ** k, t + L / 2 ** k]) for k in range(1, 9)]
        ok3 &= all(ck[i + 1] <= ck[i] + 1e-20 for i in range(len(ck) - 1))
    oks.append(ok3); msgs.append(f"3-3 [증명-수치확인]={ok3}")
    report(3, all(oks), "; ".join(msgs))


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
