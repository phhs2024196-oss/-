"""광운대 2025학년도 자연 1교시(KW-G25A) 검산 — 기대값은 V.ans() 로만 읽는다."""
import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-G25A", "items/KW/PAST")


def eq(s):
    """'A=B' 문자열 → A-B 식(sympify)."""
    l, r = s.replace(" ", "").split("=")
    loc = {"x": x, "y": y}
    return sp.sympify(l, locals=loc) - sp.sympify(r, locals=loc)


def p01():
    # [1](1) 외접원: 세 점을 지나는 원을 연립으로, 내접원: 내심 = (aA+bB+cC)/(a+b+c)
    X, Y, c1, c2, c3 = sp.symbols("X Y c1 c2 c3")
    pts = [(0, 0), (8, 0), (0, 6)]
    sol = sp.solve([xx**2 + yy**2 + c1*xx + c2*yy + c3 for xx, yy in pts], [c1, c2, c3])
    circ = x**2 + y**2 + sol[c1]*x + sol[c2]*y + sol[c3]
    O, A, B = map(sp.Matrix, pts)
    a_, b_, c_ = (B - A).norm(), (A - O).norm(), (O - B).norm()   # 각 꼭짓점 O,A,B 의 대변: a=AB, b=OB?, 아래에서 정확히 지정
    la, lb, lc = (A - B).norm(), (O - B).norm(), (O - A).norm()    # O의 대변=AB, A의 대변=OB, B의 대변=OA
    I = (la*O + lb*A + lc*B) / (la + lb + lc)
    r = sp.Rational(2*sp.Rational(1, 2)*8*6) / (la + lb + lc)      # r = 2*넓이/둘레
    inc = (x - I[0])**2 + (y - I[1])**2 - r**2
    V.check(1, "1-1 외접원", sp.expand(circ), sp.expand(eq(V.ans(1, "1-1", "외접원"))))
    V.check(1, "1-1 내접원", sp.expand(inc), sp.expand(eq(V.ans(1, "1-1", "내접원"))))
    # 외접원의 중심·반지름 = (4,3), 5 이고 내심 (2,2), r=2
    V.check(1, "1-1 외심x", -sol[c1]/2, 4); V.check(1, "1-1 내심", I[0] + I[1], 4)
    # [1](2) 두 원 위 점 사이 거리의 최솟값·최댓값(수치 최적화 — 각도 격자 + 국소 정밀화)
    Rc, Oc = 5.0, (4.0, 3.0); ri, Ic = 2.0, (2.0, 2.0)
    def dist(u, v):
        return math.hypot(Oc[0] + Rc*math.cos(u) - Ic[0] - ri*math.cos(v), Oc[1] + Rc*math.sin(u) - Ic[1] - ri*math.sin(v))
    N = 720
    grid = [(dist(2*math.pi*i/N, 2*math.pi*j/N), 2*math.pi*i/N, 2*math.pi*j/N) for i in range(N) for j in range(N)]
    def refine(best, sign):
        _, u, v = best; h = 2*math.pi/N
        for _ in range(60):
            cands = [(sign*dist(u + du*h, v + dv*h), u + du*h, v + dv*h) for du in (-1, 0, 1) for dv in (-1, 0, 1)]
            m = min(cands)
            _, u, v = m; h /= 1.6
        return dist(u, v)
    dmin = refine(min(grid), 1); dmax = refine(max(grid), -1)
    V.check(1, "1-2 최솟값", dmin, sp.N(S(V.ans(1, "1-2", "min"))), 1e-8)
    V.check(1, "1-2 최댓값", dmax, sp.N(S(V.ans(1, "1-2", "max"))), 1e-8)
    # 정확값: 두 중심 사이 거리 d, R-d-r, R+d+r
    d = sp.sqrt((4 - 2)**2 + (3 - 2)**2)
    V.check(1, "1-2 정확값(최소)", 5 - d - 2, V.ans(1, "1-2", "min")); V.check(1, "1-2 정확값(최대)", 5 + d + 2, V.ans(1, "1-2", "max"))
    # [2] 좌석 배정 전수: 학생 인덱스 0..6, 학년·성별
    grade = [1, 1, 3, 1, 2, 1, 1]; sex = ["f", "f", "m", "f", "f", "m", "m"]
    perms = list(itertools.permutations(range(7)))      # perm[seat]=student
    c1_ = sum(1 for p in perms if set(p[:3]) == {0, 1, 2})
    V.check(1, "2-1 전수", c1_, V.ans(1, "2-1"))
    def pos(p, s): return p.index(s)
    c2_ = 0
    for p in perms:
        i3, i2 = pos(p, 2), pos(p, 4)
        lo, hi = sorted((i3, i2))
        mid = p[lo + 1:hi]
        if len(mid) == 4 and sorted((sex[s], grade[s]) for s in mid) == [("f", 1), ("f", 1), ("m", 1), ("m", 1)]:
            c2_ += 1
    V.check(1, "2-2 전수", c2_, V.ans(1, "2-2"))
    def ok_nb(p, s):
        i = pos(p, s)
        if i == 0 or i == 6: return False
        a, b = p[i - 1], p[i + 1]
        return {(sex[a], grade[a]), (sex[b], grade[b])} == {("f", 1), ("m", 1)}
    c3_ = sum(1 for p in perms if ok_nb(p, 2) and ok_nb(p, 4))
    V.check(1, "2-3 전수", c3_, V.ans(1, "2-3"))


def p02():
    Pn = 2025*2026*2027*2028 - 24
    Dn = 2025**2 + 3*2023 + 1
    V.check(2, "1 나머지(정수)", Pn % Dn, V.ans(2, "1"))
    # 다항식 나눗셈으로도: x=2025
    q, r = sp.div(x*(x+1)*(x+2)*(x+3) - 24, x**2 + 3*(x - 2) + 1, x)
    V.check(2, "1 다항식 나머지", r, 11)
    # [2] 항의 단순화와 극한(정밀 수치 + 정적분)
    kk = sp.symbols("kk", integer=True, positive=True)
    num = sp.summation(3*sp.Symbol("pp", integer=True)**2 - 3*sp.Symbol("pp", integer=True) + 1, (sp.Symbol("pp", integer=True), 1, kk))
    den = sp.summation(2*sp.Symbol("pp", integer=True) - 1, (sp.Symbol("pp", integer=True), 1, kk))
    V.check(2, "2 분자=k^3", sp.simplify(num - kk**3), 0); V.check(2, "2 분모=k^2", sp.simplify(den - kk**2), 0)
    def term_sum(N):
        return sum(sp.Rational(k_, N**2 + k_**2) for k_ in range(1, N + 1))
    # 직접 정의대로 (분자·분모를 합으로 계산해) 한 항 확인
    def term_direct(N, k_):
        return sp.Rational(sum(3*p*p - 3*p + 1 for p in range(1, k_ + 1)), sum(2*p - 1 for p in range(1, k_ + 1)) * (N**2 + k_**2))
    V.check(2, "2 항 일치", sum(term_direct(30, k_) for k_ in range(1, 31)), term_sum(30))
    val = math.fsum(k_/(20000**2 + k_**2) for k_ in range(1, 20001))
    V.check(2, "2 극한(n=20000)", val, sp.N(S(V.ans(2, "2")), 12), 1e-4)
    V.check(2, "2 정적분", sp.integrate(x/(1 + x**2), (x, 0, 1)), V.ans(2, "2"))
    # [3] n 번 합성 후 넓이
    def fx(v): return v if v >= 1 else 2 - v
    ok_f = all(abs(fx(fx(v)) - fx(v)) < 1e-12 for v in [-3 + i*0.01 for i in range(700)])
    V.report(2, ok_f, "3 f∘f=f (격자 700점)")
    found = None
    for nn in range(1, 12):
        gn = x
        for _ in range(nn): gn = (gn - 1)**2 + 1
        gn_s = sp.expand(gn)
        area = sp.integrate(2 - x - gn_s, (x, 0, 1)) * 2     # [0,1] 에서 f^n = 2-x 가 위, [1,2] 대칭
        if area == sp.Rational(15, 17): found = nn
    V.check(2, "3 n", found, V.ans(2, "3"))
    # [4](1)
    fexpr_l = 2 - (x + 2)**2        # |f| 왼쪽 조각 (x<=-1)
    fexpr_r = -(-2*x - 1)           # |f|=|−2x−1|; −1<x<−1/2 에서 −2x−1>0 이므로 −2x−1 ... 아래에서 부호 확인
    assert (-2*(-0.9) - 1) > 0
    right = -2*x - 1
    h = sp.symbols("h", real=True)
    cont = (sp.limit(fexpr_l, x, -1, "-") == sp.limit(right, x, -1, "+") == abs((-1 + 2)**2 - 2))
    dl = sp.limit((fexpr_l.subs(x, -1 + h) - 1)/h, h, 0, "-")
    dr = sp.limit((right.subs(x, -1 + h) - 1)/h, h, 0, "+")
    got = f"continuous={'yes' if cont else 'no'}; differentiable={'yes' if (cont and dl == dr) else 'no'}"
    V.report(2, got == V.ans(2, "4-1"), f"4-1 {got} (좌 {dl}, 우 {dr}) / YAML {V.ans(2, '4-1')}")
    # [4](2): 역함수를 조각마다 풀어 만든 뒤 후보점 전부 검사
    yy = sp.symbols("yy", real=True)
    inv1 = [s for s in sp.solve(sp.Eq((yy + 2)**2 - 2, x), yy) if True]            # f1 의 역: yy=-2±sqrt(x+2), 정의역 -2<yy<=-1 → '+'
    inv1 = -2 + sp.sqrt(x + 2)
    inv2 = sp.solve(sp.Eq(-2*yy - 1, x), yy)[0]
    inv3 = sp.solve(sp.Eq(sp.sqrt(yy/2) + 1, x), yy)[0]
    inv3 = sp.simplify(inv3)
    # 값 확인(역함수 성질): f(inv)=x
    V.check(2, "4-2 역함수1", ((inv1 + 2)**2 - 2).subs(x, sp.Rational(-3, 2)), sp.Rational(-3, 2))
    V.check(2, "4-2 역함수3", (sp.sqrt(inv3/2) + 1).subs(x, sp.Rational(3, 2)), sp.Rational(3, 2))
    def finv_lim(a):
        # (좌극한, 우극한) — 구간 경계에서만 값이 갈리는 -1, 1
        a = sp.nsimplify(a)
        def piece(u):
            if u < -1: return inv1
            if u < 1: return inv2
            return inv3
        L = sp.limit(piece(a - sp.Rational(1, 10**6)).subs(x, a), x, a) if False else None
        # 좌우극한은 a 의 좌우 조각 식에 a 를 대입해서(연속 조각) 얻는다
        left = inv1 if a <= -1 else (inv2 if a <= 1 else inv3)
        right = inv1 if a < -1 else (inv2 if a < 1 else inv3)
        return left.subs(x, a), right.subs(x, a)
    cands = set()
    # 조각마다 해: |L+R|=1 을 풀어 그 조각 안의 해만 채택 + 경계점 -1, 1
    for lo, hi, expr in [(-2, -1, 2*inv1), (-1, 1, 2*inv2), (1, 2, 2*inv3)]:
        for sgn in (1, -1):
            for r in sp.solve(sp.Eq(expr, sgn), x):
                r = sp.nsimplify(r)
                if r.is_real and lo < r < hi: cands.add(r)
    for b in (-1, 1):
        L, R = finv_lim(b)
        if abs(L + R) == 1: cands.add(sp.Integer(b))
    mine = sorted(cands)
    exp = sorted(S(v) for v in V.ans(2, "4-2").replace(" ", "").split(";"))
    V.check(2, "4-2 해집합", mine, exp)


if __name__ == "__main__": V.run(globals())
