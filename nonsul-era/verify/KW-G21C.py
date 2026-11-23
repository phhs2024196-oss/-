import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, math, random
V = Verifier("KW-G21C", "items/KW/PAST")
R_ = sp.Rational
a, b = sp.symbols("a b", real=True)

def p01():
    # [1](1) 켤레복소수 곱: 기호 전개
    p, q, r, s = sp.symbols("p q r s", real=True)
    al, be = p + sp.I*q, r + sp.I*s
    V.check(1, "1-1 conj(ab)=conj(a)conj(b)", sp.simplify(sp.expand(sp.conjugate(al*be) - sp.conjugate(al)*sp.conjugate(be))), 0)
    # [1](2),(3): 허근 w, a 에 따라 계산
    def omega(av):
        D = (av+1)**2 - 4*(2*av+1)
        return None if D >= 0 else ((av+1) + sp.I*sp.sqrt(-D))/2
    ys = sp.solve(sp.Eq(((a+1)**2 - 2*(2*a+1)), 0), a)   # w^2+wbar^2=0
    ok_vals = [v for v in ys if sp.N((v+1)**2 - 4*(2*v+1)) < 0]
    got = sorted(ok_vals, key=lambda z: sp.N(z))
    exp = sorted(sp.sympify(V.ans(1, "1-2", "values").replace(",", " ").split()[i]) for i in range(2))
    V.check(1, "1-2 a 값", got, exp)
    for v in exp:
        w = omega(v); V.check(1, f"1-2 w^4-wbar^4 (a={v})", sp.simplify(sp.expand(w**4 - sp.conjugate(w)**4)), 0)
    # 다른 a (예: 0, 2) 에서는 0 이 아님
    V.report(1, all(sp.simplify(omega(v)**4 - sp.conjugate(omega(v))**4) != 0 for v in [0, 2, 3, -R_(1,4)]), "1-2 다른 a 에서는 w^4-wbar^4≠0")
    # 1-3 부등식: (a-1)(a-5)>0 와 허근 조건 D<0
    lo1, hi1, lo2, hi2 = (sp.sympify(V.ans(1, "1-3", k_)) for k_ in ("lo1", "hi1", "lo2", "hi2"))
    def in_ans(av): return (lo1 < av < hi1) or (lo2 < av < hi2)
    good = True
    for i in range(-300, 1000):
        av = R_(i, 100)
        w = omega(av)
        truth = (w is not None) and sp.re(sp.N(w**2 + sp.conjugate(w)**2 - 2*w*sp.conjugate(w))) > -8
        good &= (truth == in_ans(sp.N(av)))
    V.report(1, good, "1-3 범위 수치 대조(-3<a<10, 간격 0.01)")
    # [2] 정2n각형
    def dist_dist(n):
        pts = [(sp.cos(2*sp.pi*j/(2*n)), sp.sin(2*sp.pi*j/(2*n))) for j in range(2*n)]
        d2 = {}
        for i, j in itertools.combinations(range(2*n), 2):
            val = sp.nsimplify(sp.simplify((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2))
            d2[val] = d2.get(val, 0) + 1
        tot = sum(d2.values())
        return {k_: R_(v, tot) for k_, v in d2.items()}, sum(k_*R_(v, tot) for k_, v in d2.items())
    d3, e3 = dist_dist(3)
    V.check(1, "2-1 E(X3)", sp.simplify(e3), V.ans(1, "2-1", "E"))
    V.report(1, all(d3[v] == sp.sympify(V.ans(1, "2-1", key)) for v, key in ((1, "p1"), (3, "p3"), (4, "p4"))) and len(d3) == 3, "2-1 분포 {1:2/5, 3:2/5, 4:1/5}")
    d4, e4 = dist_dist(4)
    V.check(1, "2-2 E(X4)", sp.simplify(e4), V.ans(1, "2-2"))
    # 2-3: E(X_n) 식 (YAML) 과 직접 계산
    for n_ in (2, 3, 4, 5):
        _, en = dist_dist(n_)
        form = sp.sympify(V.ans(1, "2-3", "E").replace("Sum", "Sum"), locals={"n": n_, "k": sp.Symbol("k", integer=True)})
        V.check(1, f"2-3 E(X_{n_})", sp.N(sp.simplify(en), 20), sp.N(form.doit() if hasattr(form, "doit") else form, 20), tol=1e-9)
    # 2-4 극한
    nn = sp.symbols("nn", integer=True, positive=True)
    En = lambda m: 4/(2*m-1)*sum(1 - sp.cos(sp.pi*kk/m) for kk in range(1, m)) + R_(4, 2*m-1)
    V.report(1, abs(sp.N(En(2000)) - sp.sympify(V.ans(1, "2-4"))) < 1e-3, f"2-4 E(X_2000)≈{sp.N(En(2000), 8)} → 2")
    V.check(1, "2-4 적분식", sp.Rational(2)/sp.pi*integral(1 - sp.cos(x), x, 0, sp.pi), V.ans(1, "2-4"))

def p02():
    k_ = sp.symbols("k_", real=True)
    # [1](1): ① 이 1개 실근: k=b (a!=0) 또는 k!=b & D/4=0
    # a != 0 필요: a=0 이면 k=b -> 2=0 없음, k!=b -> x^2=-2/(k-b) 실근 0 또는 2개
    for av in [R_(1,2), 1, 2]:
        for bv in [R_(1,4), -1]:
            roots = sp.solve(av**2*k_**2 - 2*k_ + 2*bv, k_)
            real_roots = [r for r in roots if r.is_real and r != 0 and r != bv]
            cand = real_roots + [bv]
            # 각 후보에서 정말 실근이 1개인지 확인
            def nreal(kv):
                poly = sp.Poly((kv-bv)*x**2 + 2*av*kv*x + 2, x)
                return len({r for r in sp.roots(poly, multiple=False) if sp.im(r) == 0}) if poly.degree() > 0 else (0)
            ok = all(nreal(c) == 1 for c in cand)
            if len(roots) == 2 and all(r.is_real for r in roots):
                s_ = sum(cand)
                ex = sp.sympify(V.ans(2, "1-1", "ksum"), locals={"a": av, "b": bv})
                V.check(2, f"1-1 k 합 (a={av},b={bv})", sp.simplify(s_), ex)
                V.report(2, ok, f"1-1 후보 k 에서 실근 1개 (a={av},b={bv})")
    # a=0 이면 실근 1개 불가 (k!=b: x^2=-2/(k-b) 는 0개 또는 2개)
    V.report(2, all(len(sp.solve((kv-1)*x**2 + 2, x)) in (0, 2) for kv in [R_(1,2), 2, 3, -1]) , "1-1 a=0 이면 1개 실근 불가")
    # [1](2): 해집합(복소수 범위) 비교
    xc = sp.Symbol("xc")   # 복소수 범위 해집합
    def sol_set(aa, kk, bb, which):
        poly = (kk-bb)*xc**2 + 2*aa*kk*xc + 2 if which == 1 else 2*xc**2 + 2*aa*kk*xc + kk - bb
        return set(sp.solve(poly, xc)) if poly != 0 else None
    def same_set(aa, kk, bb):
        s1, s2 = sol_set(aa, kk, bb, 1), sol_set(aa, kk, bb, 2)
        return s1 == s2
    def cond(aa, kk, bb):
        return (kk - bb == 2) or (aa == 0 and kk - bb == -2)
    good = True
    for aa in [0, 1, R_(1,2), -2]:
        for bb in [1, -3, R_(1,3)]:
            for dd in [2, -2, 1, -1, 3, 0, R_(1,2), 4]:
                kk = bb + dd
                if kk == 0: continue
                good &= (same_set(aa, kk, bb) == cond(aa, kk, bb))
    V.report(2, good, "1-2 해집합 일치 ⇔ (k-b=2 또는 a=0,k-b=-2) 수치 대조(복소수 범위)")
    # [2] 순서쌍 개수
    cnt = [(A, B) for A in range(1, 400) for B in range(1, 400) if abs(math.log2(A) - math.log2(10)) + math.log2(B) <= 1 + 1e-12]
    V.check(2, "2 개수", len(cnt), V.ans(2, "2"))
    # [3]
    fx = x**2 + x + 1
    X, Y = sp.symbols("X Y", real=True)
    V.check(2, "3-2 f 가 조건 만족 (차이=(xy)^2>=0)", sp.expand(fx.subs(x, X+Y) - (fx.subs(x, X) + fx.subs(x, Y) - (X*Y-1)**2)), (X*Y)**2)
    V.check(2, "3-2 f'(x)", sp.diff(fx, x), sp.sympify(V.ans(2, "3-2", "fprime"), locals={"x": x}))
    V.check(2, "3-2 f(0), f'(0)", (fx.subs(x, 0), sp.diff(fx, x).subs(x, 0)), (V.ans(2, "3-1"), 1))
    V.check(2, "3-2 f", fx, sp.sympify(V.ans(2, "3-2", "f"), locals={"x": x}))
    # 3-3
    g = x*sp.exp(-x)
    a_ = sp.solve(sp.diff(g, x), x)[0]
    V.check(2, "3-3 a (극대)", a_, 1); V.report(2, sp.diff(g, x, 2).subs(x, a_) < 0, "3-3 x=a 에서 극대(g''<0)")
    V.check(2, "3-3 넓이", integral(fx - g, x, 0, a_), sp.sympify(V.ans(2, "3-3"), locals={"exp": sp.exp}))

if __name__ == "__main__": V.run(globals())
