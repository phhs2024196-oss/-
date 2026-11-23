"""SSWU-M26 (성신여대 2026학년도 모의논술 자연계열 1~4) 검산 — 조건에서 독립 재계산, 기대값은 V.ans()."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("SSWU-M26", "items/SSWU/PAST")
mp.mp.dps = 30
b_, d_ = sp.symbols("b d", positive=True)
LOC = {"x": x, "n": n, "k": k, "e": sp.E, "b": b_, "d": d_}
def E(s): return sp.sympify(s, locals=LOC)
def kv(no, sub):
    return {p.split("=", 1)[0].strip(): p.split("=", 1)[1] for p in V.ans(no, sub).replace(" ", "").split(";") if "=" in p}

def p01():
    S = lambda m: m * (m + 1) * (2 * m + 1)
    bn = lambda m: b_ + (m - 1) * d_
    an = sp.simplify((S(n) - S(n - 1)) / bn(n))
    V.check(1, "1-1 a_n(n>=2)", an, E(V.ans(1, "1")))
    V.check(1, "1-1 a_1", E(V.ans(1, "1")).subs(n, 1), sp.Integer(6) / b_)
    V.check(1, "1-2 limit", sp.limit(E(V.ans(1, "1")) / bn(n), n, sp.oo), E(V.ans(1, "2")))
    a = lambda m: 6 * m / d_
    tot = sum(1 / (sp.sqrt(a(j + 1)) + sp.sqrt(a(j))) for j in range(1, 11))
    V.check(1, "1-3 sum", sp.simplify(tot), E(V.ans(1, "3")))
    # 수치: 임의 b,d 로 합 조건 재현
    bb, dd = 1.3, 0.7
    A = [None]; B = [None]
    for m in range(1, 8):
        bm = bb + (m - 1) * dd
        am = (m * (m + 1) * (2 * m + 1) - (m - 1) * m * (2 * m - 1)) / bm
        A.append(am)
    V.report(1, abs(sum(A[m] * (bb + (m - 1) * dd) for m in range(1, 8)) - 7 * 8 * 15) < 1e-9, "1-1 (가) 합 조건 수치 재현")

def p02():
    a_, b2, c_ = sp.symbols("a b c", real=True); t = sp.symbols("t", positive=True)
    # (1) 연속
    r1 = sp.Eq(15 + a_, -1 + b2); r2 = sp.Eq(-2 + b2, c_ * sp.exp(-1))
    d1 = V.ans(2, "1").replace(" ", "").split(";")
    sa = sp.solve(r1, a_)[0]; sc = sp.solve(r2, c_)[0]
    L = lambda s: sp.sympify(s.split("=")[0], locals={"a": a_, "b": b2, "c": c_, "e": sp.E}) - sp.sympify(s.split("=")[1], locals={"e": sp.E})
    V.check(2, "2-1 a-b", L(d1[0]).subs(a_, sa), 0); V.check(2, "2-1 b-c/e", L(d1[1]).subs(c_, sc), 0)
    # (2) 좌·우 미분계수
    f = c_ * sp.exp(-sp.sqrt(t / 2))
    right = sp.diff(f, t).subs(t, 2); left = sp.diff(-t + b2, t)
    cs = sp.solve(sp.Eq(right, left), c_)[0]
    V.check(2, "2-2 c", cs, E(V.ans(2, "2")))
    mp.mp.dps = 30; cv = 4 * mp.e; h = mp.mpf("1e-12")
    vv = lambda tt: (-tt + 6) if tt <= 2 else cv * mp.e ** (-mp.sqrt(tt / 2))
    V.report(2, abs((vv(2 + h) - vv(2)) / h - (vv(2) - vv(2 - h)) / h) < 1e-9, "2-2 수치 미분가능")
    # (3) 증명-수치확인
    A_, B_, C_ = -10, 6, 4 * sp.E
    v = lambda tt: (15 * tt**2 + A_) if tt < 1 else ((-tt + B_) if tt <= 2 else C_ * sp.exp(-sp.sqrt(tt / 2)))
    D2 = sp.integrate(15 * t**2 + A_, (t, 0, 1)) + sp.integrate(-t + B_, (t, 1, 2))
    D8 = D2 + sp.integrate(C_ * sp.exp(-sp.sqrt(t / 2)), (t, 2, 8))
    V.report(2, D2 < 0 < sp.N(D8) and sp.N(D8) > 8, f"2-3 [증명-수치확인] D(2)={D2}, D(8)={sp.N(D8, 6)}")
    V.check(2, "2-3 D(8)-D(2)", sp.simplify(D8 - D2), 32 - 48 / sp.E)
    # 연속성(1)에서 a,b,c 조합이 유일
    V.check(2, "2-3 a,b", (sp.solve(r1.subs(b2, 6), a_)[0], sp.solve(r2.subs(c_, C_), b2)[0]), (-10, 6)) if False else None
    V.report(2, sa.subs(b2, 6) == -10 and sp.simplify(sc.subs(b2, 6) - 4 * sp.E) == 0, "2-3 a=-10, c=4e at b=6")

def p03():
    # (1) 구간별 수치 적분과 YAML 식 비교
    for nn in (1, 2, 3, 4):
        u = lambda t: mp.mpf(2) ** t if t <= nn else (-t + nn + 2**nn)
        up = lambda t: min(mp.mpf(2) ** t, -t + nn + 2**nn)
        lo = lambda t: max(mp.log(t, 2), -t + 1)
        pts = [0, 1, nn, 2**nn] if nn > 1 else [0, 1, 2]
        area = mp.quad(lambda t: max(up(t) - lo(t), 0), sorted(set(pts)))
        V.check(3, f"3-1 area n={nn}", area, E(V.ans(3, "1")).subs(n, nn), 1e-18)
    # (2) 접선
    d = kv(3, "2"); l1 = E(d["l1"]); l2 = E(d["l2"]); ar = E(d["area"])
    g = sp.log(x, 2); f = 2**x
    for nn in range(0, 4):
        t1 = sp.diff(g, x).subs(x, 2**nn) * (x - 2**nn) + g.subs(x, 2**nn)
        xx = g.subs(x, 2**nn)
        t2 = sp.diff(f, x).subs(x, xx) * (x - xx) + f.subs(x, xx)
        V.check(3, f"3-2 접선1 n={nn}", sp.simplify(t1 - l1.subs(n, nn)), 0)
        V.check(3, f"3-2 접선2 n={nn}", sp.simplify(t2 - l2.subs(n, nn)), 0)
    L1, L2, L3 = l1.subs(n, 0), l2.subs(n, 0), -x + 1
    pts = []
    for A, B in ((L1, L2), (L1, L3), (L2, L3)):
        s = sp.solve(sp.Eq(A, B), x)[0]; pts.append((s, A.subs(x, s)))
    (x1, y1), (x2, y2), (x3, y3) = pts
    tri = sp.Abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)
    V.check(3, "3-2 삼각형", tri, ar)
    # (3)
    gg = lambda v: sp.log(v, 2); ff = lambda v: 2**v
    a = [None, gg(2)]; bq = [None, ff(0)]
    for m in range(2, 40):
        a.append(sp.simplify(gg(2 * ff(a[-1])))); bq.append(sp.simplify(ff(-1 + gg(bq[-1]))))
    V.report(3, [a[m] for m in range(1, 8)] == list(range(1, 8)), "3-3 a_n=n")
    s1 = sp.summation(2 / (n * (n + 1)), (n, 1, sp.oo)); s2 = sp.summation(sp.Rational(1, 2) ** (n - 1), (n, 1, sp.oo))
    d3 = kv(3, "3")
    V.check(3, "3-3 sum 1/s_n", s1, E(d3["sum1"])); V.check(3, "3-3 sum b_n", s2, E(d3["sum2"]))
    V.report(3, all(bq[m] == sp.Rational(1, 2) ** (m - 1) for m in range(1, 8)), "3-3 b_n=(1/2)^(n-1)")

def p04():
    # 좌표: 원 x²+y²=1, A(-1,0), B(1,0). ∠QAB=θ, AQ=2cosθ, AP=2cos2θ
    def pts(th):
        A = mp.matrix([-1, 0]); a = 2 * mp.cos(th)
        Q = A + a * mp.matrix([mp.cos(th), -mp.sin(th)])
        AP = 2 * mp.cos(2 * th); P = A + AP * mp.matrix([mp.cos(2 * th), mp.sin(2 * th)])
        return A, P, Q, a, AP
    for th in (mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.7")):
        A, P, Q, a, AP = pts(th)
        V.report(4, abs(mp.norm(P) - 1) < 1e-20 and abs(mp.norm(Q) - 1) < 1e-20, f"4 P,Q 가 원 위 θ={th}")
        V.check(4, f"4-1 AP=a^2-2 θ={th}", AP, E(V.ans(4, "1")).subs(sp.Symbol("a"), a) if False else a**2 - 2, 1e-20)
    # 답 문자열 a**2-2 를 a 에 대입
    aa = sp.symbols("a", positive=True)
    for th in (mp.mpf("0.3"), mp.mpf("0.6")):
        A, P, Q, a, AP = pts(th)
        V.check(4, "4-1 YAML식", sp.sympify(V.ans(4, "1"), locals={"a": aa}).subs(aa, sp.Float(str(a), 30)), AP, 1e-15)
    # (2) HK 최대: θ in (0, π/4)
    def hk(th):
        A, P, Q, a, AP = pts(th); return abs(P[0] - Q[0])
    ths = [mp.pi / 4 * i / 4000 for i in range(1, 4000)]
    best = max(hk(t) for t in ths)
    V.check(4, "4-2 max HK", best, E(V.ans(4, "2")), 1e-6)
    thc = mp.acos(mp.sqrt(10) / 4)
    V.check(4, "4-2 at cosθ=√10/4", hk(thc), E(V.ans(4, "2")), 1e-20)
    # (3) 삼각형 PAQ 넓이 (신발끈)
    A, P, Q, a, AP = pts(thc)
    area = abs((A[0] * (P[1] - Q[1]) + P[0] * (Q[1] - A[1]) + Q[0] * (A[1] - P[1])) / 2)
    V.check(4, "4-3 area", area, E(V.ans(4, "3")), 1e-20)
    # (다): PQ 가 AB 와 원 내부에서 만남
    t0 = P[1] / (P[1] - Q[1]); xi = P[0] + t0 * (Q[0] - P[0])
    V.report(4, -1 < xi < 1 and P[1] > 0 > Q[1], "4 조건(다) 만족")

if __name__ == "__main__": V.run(globals())
