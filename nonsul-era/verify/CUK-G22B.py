#!/usr/bin/env python3
"""CUK-G22B (가톨릭대 2022 수시 논술 의예 1·3·4; 의예 2 는 가이드북에 없음) — 문제 조건에서 독립 재계산."""
import sys, pathlib, itertools, random
from fractions import Fraction as Fr
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G22B", "items/CUK/PAST")


def p01():
    # 주사위 3개(서로 다른 주사위 = 순서 있는 216가지)로 상자 결정, 이후 공 뽑기를 정확히 열거
    pa = Fr(sum(1 for d in itertools.product(range(1, 7), repeat=3) if max(d) - min(d) in (2, 4)), 216)
    box0 = {"A": (1, 4), "B": (3, 2)}                       # (검은 공, 흰 공)
    prob = {"A": pa, "B": 1 - pa}
    p_black = Fr(0); p_w_b = Fr(0); p_w = Fr(0)
    for b1 in "AB":
        bl, wh = box0[b1]
        # 첫 번째: 흰 공
        pw = prob[b1] * Fr(wh, bl + wh)
        pb = prob[b1] * Fr(bl, bl + wh)
        p_black += pb
        p_w += pw
        for b2 in "AB":
            bl2, wh2 = box0[b2]
            if b2 == b1:
                wh2 -= 1                                    # 같은 상자에서 흰 공을 이미 뽑았음(비복원)
            p_w_b += pw * prob[b2] * Fr(bl2, bl2 + wh2)
    q = p_w_b / p_w
    exp = dict(s.split("=", 1) for s in V.ans(1, "1").replace(" ", "").split(";"))
    V.check(1, "상자 A 선택확률", pa, Fr(4, 9))
    V.check(1, "p", p_black, Fr(exp["p"]))
    V.check(1, "q", q, Fr(exp["q"]))


def p03():
    # a_k: 원순열을 직접 열거 (k+2 명 중 A,B 포함 5명, 회전 동일시, A-B 이웃)
    def a_brute(k):
        others = list(range(2, k + 2))
        tot = 0
        for pick in itertools.combinations(others, 3):
            people = (0, 1) + pick
            seen = set()
            for perm in itertools.permutations(people):
                i = perm.index(0)
                rot = perm[i:] + perm[:i]                   # 회전 정규화(0 을 맨 앞으로)
                seen.add(rot)
            for rot in seen:
                if rot.index(1) in (1, 4):                  # A=0 의 이웃은 위치 1 과 4
                    tot += 1
        return tot
    ok = all(a_brute(k) == 2 * k * (k - 1) * (k - 2) for k in range(3, 8))
    V.report(3, ok, "a_k=2k(k-1)(k-2) (k=3..7 원순열 직접 열거)")
    # 소수 p 마다 L=Σ b_k 를 정의대로 계산(b_k = ceil(a_k/p)), YAML 식과 비교
    pp = sp.symbols("p")
    expr = sp.sympify(V.ans(3, "1"), locals={"p": pp})
    bad = []
    for pr in sp.primerange(5, 120):
        a = lambda k: 2 * k * (k - 1) * (k - 2)
        L = sum(-(-a(k) // pr) for k in range(3, pr))       # 올림 = 가장 작은 자연수 m >= a_k/p
        if L != expr.subs(pp, pr):
            bad.append(pr)
    V.proof(3, "L 공식(소수 5~113)", bad)


def p04():
    mp.mp.dps = 50
    x = sp.symbols("x")

    def mk(al, be, kk):
        # f(x)=k(x-(3al-be)/2)(x-be)^2 : x=al 에서 극댓값, x=be 에서 극솟값 0 (k>0) — f' 와 f(be) 로 확인
        f = lambda X: kk * (X - (3 * al - be) / sp.Integer(2)) * (X - be) ** 2
        assert sp.simplify(sp.diff(f(x), x) - 3 * kk * (x - al) * (x - be)) == 0 and f(be) == 0
        P = sp.Poly(sp.expand(f(x - f(x)) - f(x)), x)
        return f, sp.Poly(sp.sqf_part(P.as_expr()), x)         # 중근 제거(서로 다른 실근만)

    def real_roots(Q):
        r = mp.polyroots([mp.mpf(str(sp.N(c, 50))) for c in Q.all_coeffs()], maxsteps=2000, extraprec=300)
        return sorted(z.real for z in r if abs(mp.im(z)) < mp.mpf(10) ** -25)

    # 논제 1: 임의의 (α,β,k) 에서 열린구간 (α,β) 안의 서로 다른 실근 개수(정확한 유리수 계산)
    random.seed(1)
    cnts = set()
    for _ in range(25):
        al = sp.Rational(random.randint(-9, 9), 3)
        be = al + sp.Rational(random.randint(1, 12), 3)
        kk = sp.Rational(random.randint(1, 40), 7)
        f, Q = mk(al, be, kk)
        n_open = Q.count_roots(al, be) - (1 if Q.eval(al) == 0 else 0) - (1 if Q.eval(be) == 0 else 0)
        cnts.add(n_open)
    ans4 = dict(s.split("=") for s in V.ans(4, "1").replace(" ", "").split(";"))
    V.report(4, cnts == {int(ans4["n"])}, f"(α,β) 안 서로 다른 실근 개수 집합={cnts} / YAML n={ans4['n']}")

    # 논제 2: 평행이동·닮음변환으로 α=-1, β=0 으로 놓아도 m 은 불변 → 조건 (가)의 f(b)=f(d) 로 k 결정
    al, be = sp.Integer(-1), sp.Integer(0)

    def info(kv):
        f, Q = mk(al, be, sp.Rational(str(mp.nstr(kv, 45))))
        r = real_roots(Q)
        return f, r

    def gap(kv):
        f, r = info(kv)
        if len(r) != 5:
            return None
        fx = lambda X: mp.mpf(str(sp.N(f(sp.Float(mp.nstr(X, 45), 50)), 50)))
        return fx(r[1]) - fx(r[3])

    lo, hi = mp.mpf("3.3"), mp.mpf("3.7")
    glo, ghi = gap(lo), gap(hi)
    V.report(4, glo is not None and ghi is not None and glo * ghi < 0, f"k∈[3.3,3.7] 에서 서로 다른 실근 5개, f(b)-f(d) 부호 변화")
    for _ in range(80):
        mid = (lo + hi) / 2
        gm = gap(mid)
        if gm * glo > 0:
            lo, glo = mid, gm
        else:
            hi = mid
    kv = (lo + hi) / 2
    f, r = info(kv)
    a_, b_, c_, d_, e_ = r
    fx = lambda X: mp.mpf(str(sp.N(f(sp.Float(mp.nstr(X, 45), 50)), 50)))
    m = kv * (b_ - a_) * fx(2 * b_ - c_)
    V.report(4, abs(m - mp.mpf(ans4["m"])) < 1e-8, f"k={mp.nstr(kv,12)} a..e={[mp.nstr(z,8) for z in r]} m={mp.nstr(m,15)}")


if __name__ == "__main__":
    V.run(globals())
