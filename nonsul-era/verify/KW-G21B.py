import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from math import comb
V = Verifier("KW-G21B", "items/KW/PAST")
R_ = sp.Rational

def fl(s):
    return sorted(sp.nsimplify(p) for p in s.split(","))

def p01():
    # [1] 집합
    U = {R_(n, 24) for n in range(25)}
    A = {xx for xx in U if sp.simplify(sp.sin(4*sp.pi*xx)) == 0}
    B = {xx for xx in U if sp.simplify(sp.cos(6*sp.pi*xx)) == 0}
    # A, B 는 [0,1] 의 모든 해(U 밖이면 빠짐) — 해 전체가 U 에 들어 있는지 확인
    solA = {R_(k, 4) for k in range(0, 5)}
    solB = {R_(2*k+1, 12) for k in range(0, 6)}
    V.report(1, solA <= U and solB <= U and A == solA and B == solB, "1 A,B 가 U 의 부분집합이고 해 전체와 일치")
    V.check(1, "1-1 A∪B", sorted(A | B), fl(V.ans(1, "1-1", "union")))
    V.check(1, "1-1 A∩B", sorted(A & B), fl(V.ans(1, "1-1", "inter")))
    V.check(1, "1-2 |A^c∩B^c|", len(U - (A | B)), V.ans(1, "1-2"))
    # [2](1)
    B_ = sp.symbols("B_", real=True)
    f = 1 + sp.sin(B_) + sp.cos(B_)
    V.check(1, "2-1 max", extrema(f, B_, 0, sp.pi/2)[1], V.ans(1, "2-1"))
    # [2](2) 수치: 임의 삼각형
    import random
    random.seed(1); ok = True
    for _ in range(20):
        a_ = random.uniform(0.3, 2); b2 = random.uniform(0.3, 2)
        Aa = random.uniform(0.3, 1.2); Bb = random.uniform(0.3, 1.2)
        Cc = sp.pi - Aa - Bb
        if Cc <= 0.05: continue
        e = (sp.sin(Aa)**2 + sp.sin(Bb)**2 + sp.sin(Cc)**2 - 2*sp.sin(Bb)*sp.sin(Cc)*sp.cos(Aa) - 2*sp.sin(Cc)*sp.sin(Aa)*sp.cos(Bb) - 2*sp.sin(Aa)*sp.sin(Bb)*sp.cos(Cc))
        ok &= abs(sp.N(e)) < 1e-9
    V.report(1, ok and sp.sympify(V.ans(1, "2-2")) == 0, "2-2 식의 값 0 (임의 삼각형 20개)")
    # [3](1) 수직 기울기: 점 쌍을 만들어 확인
    fx = lambda xv: sp.sqrt(-(xv+1)) - 1
    res = set()
    for av in [R_(-10), R_(-5)]:
        # b+d=1, f(c)=d: c 를 구한다. f(x)=1-b 를 풀어 c
        bv = fx(av)
        dv = 1 - bv
        cv = -(dv+1)**2 - 1
        if dv >= -1 and cv > av and cv <= -1:
            res.add(sp.simplify(-(cv-av)/(dv-bv)))
    V.check(1, "3-1 기울기", list(res)[0] if len(res) == 1 else None, V.ans(1, "3-1"))
    # [3](2) 접하는 k, 만나는 범위
    k_ = sp.symbols("k_", real=True)
    ks = sp.solve(sp.discriminant(k_**2*x**2 + (2*k_+1)*x + 2, x), k_)
    kmin = [kv for kv in ks if kv <= 1][0]
    V.check(1, "3-2 kmin", kmin, V.ans(1, "3-2", "kmin"))
    V.check(1, "3-2 kmax", 1, V.ans(1, "3-2", "kmax"))
    # 구간 밖 k 에서 교점 없음, 구간 안(경계 포함) 교점 있음 (수치)
    def meets(kv):
        sols = sp.solve(sp.Eq(kv*x, fx(x)), x)
        return any(abs(sp.im(sp.N(s))) < 1e-9 and sp.re(sp.N(s)) <= -1 + 1e-12 for s in sols)
    kn = float(kmin); V.report(1, meets(sp.Float(kn+0.01)) and meets(sp.Float(0.9)) and meets(sp.Integer(1)) and not meets(sp.Float(1.1)) and not meets(sp.Float(kn-0.05)), "3-2 범위 안/밖 수치 확인")
    # [3](3) 넓이
    kk = (1 - sp.sqrt(2))/2
    xt = sp.simplify(-(2*kk+1)/(2*kk**2)); yt = sp.simplify(kk*xt)
    V.check(1, "3-3 접점 x", xt, -2*(2+sp.sqrt(2))); V.check(1, "3-3 접점 y", yt, sp.sqrt(2))
    yy = sp.symbols("yy", real=True)
    # 영역: 곡선 x=-(y+1)^2-1 (y>=-1), 직선 y=kk*x 와 y=x. 직선 사이 영역에서 곡선까지: y 에 대해 적분
    # y in [-1,0]: x 범위 곡선(왼쪽)~ 직선 y=x (오른쪽, x=y).  y in [0,sqrt2]: 곡선(왼쪽) ~ 직선 y=kk x (오른쪽, x=y/kk)
    xc = -(yy+1)**2 - 1
    area = integral(yy - xc, yy, -1, 0) + integral(yy/kk - xc, yy, 0, sp.sqrt(2))
    V.check(1, "3-3 넓이", sp.simplify(area), V.ans(1, "3-3"))

def p02():
    s = sum(comb(5, i)**2 for i in range(6))
    V.check(2, "1-1 합=252", s, comb(10, 5))
    ps = sorted({p + q for p in range(0, 600) for q in range(0, p+1) if comb(p, q) == s} if True else [])
    V.check(2, "1-1 p+q 목록", ps, fl(V.ans(2, "1-1", "sums")))
    V.check(2, "1-2 mod3", pow(2021, 10, 3), V.ans(2, "1-2", "mod3"))
    V.check(2, "1-2 mod7", pow(2021, 10, 7), V.ans(2, "1-2", "mod7"))
    V.check(2, "1-3 mod21", pow(2021, 10, 21), V.ans(2, "1-3"))
    # [2]
    fm = lambda xv, av: min(((xv-2)**2 if xv >= 0 else 3), av*xv + 1)
    # 2-1: a=0, h 와 x^2 사이 넓이 [-1,1]: 1 - x^2 (수치적분)
    V.check(2, "2-1 넓이", integral(1 - x**2, x, -1, 1), V.ans(2, "2-1"))
    # h(x)=min: x<1 구간 h=1, 1~3 (x-2)^2, 3~ 1  (수치 샘플 확인)
    samp = all(fm(sp.Rational(v, 4), 0) == (1 if (v/4 < 1 or v/4 > 3) else (sp.Rational(v, 4)-2)**2) for v in range(-20, 40))
    V.report(2, samp, "2-1 h(x) 구간 표현 샘플 확인")
    # 2-2 접선
    a = sp.symbols("a", real=True)
    sols = sp.solve(sp.discriminant(x**2 - (a+4)*x + 3, x), a)
    V.check(2, "2-2 a", [sv for sv in sols if sp.N(sv) > -4][0], V.ans(2, "2-2"))
    # 2-3 미분 불가능한 점 개수: h=min(f,g) 의 꺾임/불연속 점 수 — 수치로 a 별 개수
    def count_nd(av):
        av = sp.nsimplify(av)
        pts = set()
        # 교점
        if av != 0:
            r1 = 2/av
            if r1 < 0: pts.add(sp.nsimplify(r1))      # g=3, x<0
        for r in sp.solve((x-2)**2 - av*x - 1, x):
            if sp.im(r) == 0 and sp.re(r) >= 0:
                pts.add(sp.nsimplify(sp.re(r)))
        # 접하는 점은 미분 가능
        nd = 0
        for p_ in pts:
            # 좌우 기울기: 점에서 h 는 f 에서 g 로 (또는 반대로) 바뀜
            fs = 3 if p_ < 0 else (x-2)**2
            fprime = sp.diff(fs, x).subs(x, p_)
            if sp.simplify(fprime - av) != 0: nd += 1
        # f 가 x=0 에서 불연속이지만 h(0)=min(4,1)=1=g(0) 이고 근방에서 g<f 이면 미분 가능 (a>-2/... 조건은 아래 샘플)
        return nd
    def count_sample(av):
        av = sp.nsimplify(av); return count_nd(av)
    amin = -4 + 2*sp.sqrt(3)
    lo = sp.N(amin)
    cases = {"a>=0": [0, 1, 3], "amin<a<0": [sp.Float(lo+0.1), sp.Float(lo/2)], "a<amin": [sp.Float(lo-0.2), -2, -10]}
    exp = {"a>=0": 2, "amin<a<0": 3, "a<amin": 1}
    # 경계 조건은 YAML 답의 amin/amax 로
    amin_y = sp.sympify(V.ans(2, "2-3", "amin"))
    V.check(2, "2-3 amin", amin_y, amin)
    V.report(2, V.ans(2, "2-3", "amax").startswith("0"), "2-3 amax=0")
    for name, avs in cases.items():
        for av in avs:
            V.report(2, count_nd(av) == exp[name], f"2-3 {name} a={sp.N(av,4)} 개수 {count_nd(av)} (기대 {exp[name]})")
    # 3개가 되는 a 가 정확히 열린구간 (amin,0) 인지 촘촘히 확인
    good = [sp.Rational(i, 50) for i in range(-60, 20) if count_nd(sp.Rational(i, 50)) == 3]
    V.report(2, all(lo < float(g) < 0 for g in good) and len(good) == len([i for i in range(-60, 20) if lo < i/50 < 0]), "2-3 3개인 a 가 (amin,0) 과 일치")

if __name__ == "__main__": V.run(globals())
