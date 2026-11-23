"""광운대 2023학년도 자연 2교시(G23B) 문제 1·2 검산 — 기대값은 YAML(V.ans)에서만 읽는다."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-G23B", "items/KW/PAST")
a, p, q = sp.symbols("a p q", real=True)
LOC = {"x": x, "n": n, "k": k, "a": a, "p": p, "q": q}


def ex(s):
    return sp.sympify(s, locals=LOC)


def p01():
    # [1] 일대일대응 전수 열거
    A = [d for d in range(1, 7) if 6 % d == 0]
    B = [m for m in range(2, 10) if sp.isprime(m)]
    f_all = [dict(zip(A, perm)) for perm in itertools.permutations(B)]
    fs = [f for f in f_all if all(u < f[u] for u in A)]
    V.check(1, "1-1 f 개수", len(fs), V.ans(1, "1-1", "count"))
    f = fs[0]
    finv = {v: u for u, v in f.items()}
    V.check(1, "1-1 (f^-1∘f∘f^-1)(5)", finv[f[finv[5]]], V.ans(1, "1-1", "value"))
    gs = [g for g in f_all if all(u <= g[u] for u in A)]
    V.check(1, "1-2 g 개수", len(gs), V.ans(1, "1-2"))

    # [2](1) 접함 조건: 교점 중 공통접선인 점이 있는가
    def tangent_points(pv, qv):
        X, Y = sp.symbols("X Y", real=True)
        sols = sp.solve([sp.Eq(Y, -X ** 2 + pv * X + qv), sp.Eq(X, -Y ** 2 + pv * Y + qv)], [X, Y], dict=True)
        out = []
        for s in sols:
            xv, yv = s[X], s[Y]
            if not (xv.is_real and yv.is_real):
                continue
            if sp.simplify((-2 * xv + pv) * (pv - 2 * yv) - 1) == 0:
                out.append((xv, yv))
        return out

    relA, relB = ex(V.ans(1, "2-1", "rel_a")), ex(V.ans(1, "2-1", "rel_b"))
    okA = okB = okN = True
    for pv in [-3, -1, 0, 1, 2, 3]:
        qa = sp.solve(relA.subs(p, pv), q)[0]
        qb = sp.solve(relB.subs(p, pv), q)[0]
        okA &= bool(tangent_points(sp.Integer(pv), qa))
        okB &= bool(tangent_points(sp.Integer(pv), qb))
        okN &= not tangent_points(sp.Integer(pv), qa + sp.Rational(1, 3)) and not tangent_points(sp.Integer(pv), qb + sp.Rational(1, 3))
    V.report(1, okA, "2-1 관계식 rel_a 이면 접한다")
    V.report(1, okB, "2-1 관계식 rel_b 이면 접한다")
    V.report(1, okN, "2-1 관계식 둘 다 아니면(q를 1/3 옮김) 접하지 않는다")

    # [2](2) 각 자취와 그 대칭 도형의 교점 수
    X, Y = sp.symbols("X Y", real=True)

    def count(qx):  # qx: p(=X) 의 함수 q(X)
        sols = sp.solve([sp.Eq(Y, qx(X)), sp.Eq(X, qx(Y))], [X, Y], dict=True)
        return len({(s[X], s[Y]) for s in sols if s[X].is_real and s[Y].is_real})

    qa_f = lambda t: sp.solve(relA.subs(p, t), q)[0]
    qb_f = lambda t: sp.solve(relB.subs(p, t), q)[0]
    V.check(1, "2-2 자취 rel_a 교점 수", count(qa_f), V.ans(1, "2-2", "count_a"))
    V.check(1, "2-2 자취 rel_b 교점 수", count(qb_f), V.ans(1, "2-2", "count_b"))
    if V.verbose:
        # 참고: 두 자취의 합집합(및 서로의 대칭)까지 합친 교점 수 — 공식은 각각을 따로 다룸
        pts = set()
        for q1 in (qa_f, qb_f):
            for q2 in (qa_f, qb_f):
                sols = sp.solve([sp.Eq(Y, q1(X)), sp.Eq(X, q2(Y))], [X, Y], dict=True)
                pts |= {(s[X], s[Y]) for s in sols if s[X].is_real and s[Y].is_real}
        print("INFO 2-2 합집합(두 자취와 두 대칭)의 서로 다른 교점 수 =", len(pts))

    # [3](1) 치환적분 등식 수치 확인
    fns = [lambda t: mp.exp(t) * mp.cos(t), lambda t: t ** 3 - 2 * t + 1, lambda t: mp.sin(t) / (1 + t * t)]
    bad = []
    for fn in fns:
        for av in (0.7, 1.9, -1.3, 3.1):
            L = mp.quad(fn, [0, av])
            R = mp.quad(lambda t: fn(av - t), [0, av])
            if abs(L - R) > 1e-12:
                bad.append((av, L, R))
    V.proof(1, "3-1 ∫f(x)=∫f(a-x)", bad)
    # [3](2)
    mp.mp.dps = 40
    want = sp.N(ex(V.ans(1, "3-2")), 30)
    for m in (1, 3, 11, 101, 2023):
        val = mp.quad(lambda t: mp.cos(t) ** m / (mp.sin(t) ** m + mp.cos(t) ** m), [0, mp.pi / 4, mp.pi / 2])
        V.check(1, f"3-2 지수 {m}", sp.Float(str(val), 30), want, tol=1e-12)
    # [4](1)
    S = ex(V.ans(1, "4-1"))
    for av in (0.6, 1.0, 1.7):
        fpos = lambda t, av=av: -abs(t - av ** 2) * (t + av ** 2) + av ** 4
        right = mp.sqrt(2) * av ** 2
        val = mp.quad(fpos, [0, av ** 2, right])
        V.check(1, f"4-1 a={av} 넓이", sp.Float(str(val), 25), sp.N(S.subs(a, av), 25), tol=1e-12)
    # [4](2) g(a) 조각식과 극솟값
    g1, g2, g3 = (ex(V.ans(1, "4-2", key)) for key in ("g1", "g2", "g3"))
    import numpy as np
    fn = lambda t, av: -np.abs(t - av ** 2) * (t + av ** 2) + av ** 4
    bad = []
    for av in np.concatenate([np.linspace(0.05, 0.99, 40), np.linspace(1.0, 2.0, 30), np.linspace(2.01, 6, 30)]):
        tt = np.concatenate([np.linspace(av, av + 2, 200001), [av ** 2]])
        tt = tt[(tt >= av) & (tt <= av + 2)]
        mx = fn(tt, av).max()
        pc = g1 if av < 1 else (g2 if av <= 2 else g3)
        if abs(mx - float(pc.subs(a, float(av)))) > 1e-6 * max(1, abs(mx)):
            bad.append(float(av))
    V.proof(1, "4-2 g(a)=구간 최댓값", bad)
    av = np.linspace(0.02, 5, 100001)
    gv = np.where(av < 1, 2 * av ** 4 - av ** 2, np.where(av <= 2, av ** 4, (av + 2) ** 2))
    d = np.diff(gv)
    sc = [i for i in range(1, len(d)) if d[i - 1] * d[i] < 0]
    V.check(1, "4-2 극값 개수", len(sc), 1)
    V.check(1, "4-2 극솟값", sp.Float(float(gv[sc[0]]), 8), sp.N(ex(V.ans(1, "4-2", "min")), 8), tol=1e-6)


def p02():
    # [1](1)
    ints = []
    for av in range(-60, 61):
        D = 4 * av ** 2 - 4 * (4 * av + 5)
        if D < 0 and abs(av - 2) > 0 and abs(av - 2) != 1:
            ints.append(av)
    V.check(2, "1-1 정수 a", ints, [ex(V.ans(2, "1-1", "a1")), ex(V.ans(2, "1-1", "a2"))])
    # [1](2)
    y = sp.symbols("y", positive=True)
    inv = sp.solve(sp.Eq(y, sp.log(x + 5, 2)), x)[0]
    V.check(2, "1-2 역함수", sp.simplify(inv.subs(y, 3)), 2 ** 3 - 5)
    F = lambda t: 2 ** t - 5 - 6 / 2 ** t
    L = ex(V.ans(2, "1-2", "lower"))
    V.check(2, "1-2 경계에서 0", sp.simplify(F(L)), 0)
    V.report(2, F(sp.N(L) - 0.01) < 0 < F(sp.N(L) + 0.01), "1-2 경계 좌우 부호 −,+ (x>lower)")
    # [2]
    mp.mp.dps = 50
    N_ = mp.mpf(10) ** 14
    val = N_ ** (mp.mpf(1) / 3) * ((N_ ** 2 + N_) ** (mp.mpf(1) / 3) - (N_ ** 2 + 1) ** (mp.mpf(1) / 3))
    V.check(2, "2 극한(n=1e14 수치)", sp.Float(str(val), 30), sp.N(ex(V.ans(2, "2")), 30), tol=1e-6)
    # [3]
    ok = all(((kv ** 2 + (nv - kv) ** 2) == ex(V.ans(2, "3-1")).subs({k: kv, n: nv})) for nv in range(1, 8) for kv in range(0, nv + 1))
    V.report(2, ok, "3-1 거리의 제곱 식(격자점 전수)")
    ok2 = all(sp.Rational(sum(kv ** 2 + (nv - kv) ** 2 for kv in range(nv + 1)), nv + 1) == ex(V.ans(2, "3-2")).subs(n, nv) for nv in range(1, 40))
    V.report(2, ok2, "3-2 기댓값 E(X_n) (n=1..39 전수)")
    # [4](1)
    r, s, t, pp, qq = sp.symbols("r s t pp qq")
    fx = r * x ** 2 + s * x + t
    lhs = sp.expand(x * (sp.diff(fx, x) + x) * (sp.diff(fx, x) - x))
    rhs = sp.expand(fx + pp * x ** 3 + qq * x ** 2)
    eqs = sp.Poly(lhs - rhs, x).all_coeffs()
    sols = sp.solve(eqs, [r, s, t, pp], dict=True)
    # 2차 이하 일반해(r≠0 또는 1차): 미지수 r,s,t,p 를 풀어 q 가 자유로운 해와 비교
    fa, fb, fc = ex(V.ans(2, "4-1", "fa")), ex(V.ans(2, "4-1", "fb")).subs(q, qq), ex(V.ans(2, "4-1", "fc")).subs(q, qq)
    got = [sp.expand(so[r] * x ** 2 + so[s] * x + so[t]) for so in sols if so[r] != 0 or so[s] != 0]
    exp_ = [sp.expand(v) for v in (fa, fb, fc)]
    V.report(2, all(any(sp.simplify(e - g_) == 0 for g_ in got) for e in exp_[1:]) and len(got) == 2, f"4-1 2차식 일반해(q 모수) = 공식 fb, fc: {got}")
    # 공식 해가 항등식을 만족하는지 직접 대입
    def holds(fv, pv, qv):
        L_ = x * (sp.diff(fv, x) + x) * (sp.diff(fv, x) - x)
        return sp.simplify(sp.expand(L_ - (fv + pv * x ** 3 + qv * x ** 2))) == 0
    V.report(2, holds(fa, ex(V.ans(2, "4-1", "pa")), ex(V.ans(2, "4-1", "qa"))), "4-1 f=x, p=-1, q=0")
    V.report(2, holds(fb, ex(V.ans(2, "4-1", "pb")).subs(q, qq), qq), "4-1 f=-q x^2, p=4q^2-1")
    V.report(2, holds(fc, ex(V.ans(2, "4-1", "pc")).subs(q, qq), qq), "4-1 f=q x^2/3+x, p=4q^2/9-1")
    # 완전성: 3차 이상은 불가능(최고차항 비교) — 3~6차 일반 다항식에서 해 없음
    for dgr in range(3, 7):
        cs = sp.symbols(f"c0:{dgr + 1}")
        fv = sum(cs[i] * x ** i for i in range(dgr + 1))
        diff = sp.expand(x * (sp.diff(fv, x) + x) * (sp.diff(fv, x) - x) - (fv + pp * x ** 3 + qq * x ** 2))
        top = sp.Poly(diff, x).coeff_monomial(x ** (2 * dgr - 1))
        V.report(2, sp.simplify(top - cs[dgr] ** 2 * dgr ** 2) == 0 and 2 * dgr - 1 > dgr, f"4-1 {dgr}차 최고차항 {2*dgr-1}차 계수 {dgr}^2 c^2 ≠ 0")
    # [4](2)
    tot = 0
    pb, pc = ex(V.ans(2, "4-1", "pb")), ex(V.ans(2, "4-1", "pc"))
    for pexp, fexp in ((pb, ex(V.ans(2, "4-1", "fb"))), (pc, ex(V.ans(2, "4-1", "fc")))):
        for qv in sp.solve(sp.Eq(pexp, 0), q):
            tot += fexp.subs(q, qv).subs(x, 2023)
    V.check(2, "4-2 나머지 합", tot, V.ans(2, "4-2"))


if __name__ == "__main__":
    V.run(globals())
