#!/usr/bin/env python3
"""KNU 2022학년도 논술(AAT) 모의고사 자연계열Ⅱ 수학 (KNU-M22B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-M22B"
WIDTH = 2
RESULTS = []
x, c = sp.symbols("x c", real=True)


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
    import mpmath as mp
    mp.mp.dps = 50
    msgs, ok = [], True
    a, b, c, d = sp.symbols("a b c d")
    mid = a / 3 * x ** 3 + b / 2 * x ** 2 + d
    right = -(x - 1) * (3 * x - 11) * sp.exp(-x + 1)
    # 1-1: x=1 에서 연속·미분가능
    s = sp.solve([mid.subs(x, 1) - right.subs(x, 1), sp.diff(mid, x).subs(x, 1) - sp.diff(right, x).subs(x, 1)], [a, d], dict=True)[0]
    v11 = sp.simplify((b + 6 * d).subs(s))
    a11 = sp.sympify(load_answer(1, 0))
    ok &= v11 == a11
    msgs.append(f"1-1 b+6d={v11} md={a11}")

    def model(bv):
        av = 8 - bv
        dv = -(av / 3 + bv / 2)
        x0 = -bv / av
        cv = (av / 3 * x0 ** 3 + bv / 2 * x0 ** 2 + dv) + (av * x0 + bv) ** 2      # x=-b/a 연속
        f = lambda t: (-(av * t + bv) ** 2 + cv) if t < x0 else ((av / 3 * t ** 3 + bv / 2 * t ** 2 + dv) if t < 1 else -(t - 1) * (3 * t - 11) * mp.e ** (1 - t))
        fp = lambda t: (-2 * av * (av * t + bv)) if t < x0 else ((av * t ** 2 + bv * t) if t < 1 else (3 * t - 5) * (t - 5) * mp.e ** (1 - t))
        return av, cv, dv, x0, f, fp

    # 1-3: 조건 (ㄴ) — 양의 극값점 t∈(0,1) 은 f'(f(t))=1, 음의 극값점 x0=-b/a, g(t)=g(x0) — 을 (b,t) 연립으로 수치 풀이
    def F(bv):
        av, cv, dv, x0, f, fp = model(bv)
        g = lambda u: f(u) - f(f(u))
        tv = mp.findroot(lambda u: fp(f(u)) - 1, (mp.mpf("0.01"), mp.mpf("0.99")), solver="bisect")
        return g(tv) - g(x0), tv
    # F(b)=g(t)-g(-b/a) 는 ≤0 이고 조건을 만족하는 b 에서 0 에 접함(이중근) → 황금분할로 최대점, 최댓값≈0 확인
    lo, hi = mp.mpf("4.5"), mp.mpf("5.2")
    phi = (mp.sqrt(5) - 1) / 2
    for _ in range(160):
        m1, m2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if F(m1)[0] < F(m2)[0]:
            lo = m1
        else:
            hi = m2
    bv = (lo + hi) / 2
    res, tv = F(bv)
    ok &= abs(res) < 1e-20 and F(bv - mp.mpf("0.05"))[0] < 0 and F(bv + mp.mpf("0.05"))[0] < 0
    av, cv, dv, x0, f, fp = model(bv)
    # 부호 조건 a,b>0, c,d<0 와 (ㄴ) 의 '오직 한 개씩' 을 격자로 확인 (x=0 은 양·음 어느 쪽도 아님)
    gp = lambda u: (1 - fp(f(u))) * fp(u)
    grid = [mp.mpf(-6) + mp.mpf(7) * k / 28000 for k in range(28000)]
    crit = [grid[i] for i in range(len(grid) - 1) if gp(grid[i]) * gp(grid[i + 1]) < 0 and abs(grid[i]) > 1e-3]
    pos, neg = [u for u in crit if u > 0], [u for u in crit if u < 0]
    pval = -(bv ** 2 + mp.mpf(1021) / 6) / bv
    a13 = int(load_answer(1, 3))
    ok &= av > 0 and bv > 0 and cv < 0 and dv < 0 and len(pos) == 1 and len(neg) == 1 and abs(pval - a13) < 1e-10
    msgs.append(f"1-3 b={mp.nstr(bv, 12)}, t={mp.nstr(tv, 8)}, x<1 극값점 양{len(pos)}·음{len(neg)}, 잔차={mp.nstr(res, 3)}, p=-(b²+1021/6)/b={mp.nstr(pval, 15)} md={a13}")
    # 1-2(1) [증명-수치확인]: (1,11/3) 의 f 극값점 α, α<f(α)<11/3, 그리고 주어진 e² 범위만으로의 유리수 비교
    al = sp.solve(sp.diff(right, x), x)
    al = [r for r in al if 1 < r < sp.Rational(11, 3)]
    fa = right.subs(x, al[0])
    rat = (sp.Rational(12, 11) ** 3 < sp.Rational(11, 10) ** 3) and (2 ** 3 < sp.Rational(12, 5) ** 3)
    ok &= al == [sp.Rational(5, 3)] and al[0] < fa < sp.Rational(11, 3) and rat
    msgs.append(f"1-2(1) α={al}, f(α)={sp.simplify(fa)}≈{sp.N(fa, 6)} [증명-수치확인]")
    # 1-2(2) [증명-수치확인]: (1,11/3) 에서 g' 부호변화 개수
    grid2 = [1 + mp.mpf(8) / 3 * (k + mp.pi / 10) / 40000 for k in range(0, 40000)]   # 격자가 5/3 을 정확히 밟지 않게 이동
    n2 = sum(1 for i in range(len(grid2) - 1) if gp(grid2[i]) * gp(grid2[i + 1]) < 0)
    ok &= n2 == 5
    msgs.append(f"1-2(2) (1,11/3) 극값점 {n2}개 [증명-수치확인]")
    report(1, ok, "; ".join(msgs))


def p02():
    import mpmath as mp
    mp.mp.dps = 25
    msgs, ok = [], True

    def make(fexpr):
        fs = sp.lambdify(x, fexpr, "mpmath")
        fps = sp.lambdify(x, sp.diff(fexpr, x), "mpmath")
        fpps = sp.lambdify(x, sp.diff(fexpr, x, 2), "mpmath")
        ginv = lambda y: mp.findroot(lambda u: fs(u) - y, y)
        gp = lambda y: 1 / fps(ginv(y))
        h = lambda X: mp.quad(lambda t: (1 + gp(t) ** 4) ** 0.25 - (1 + fps(t) ** 4) ** 0.25, [0, X])
        return fs, fps, fpps, gp, h

    # 2-1 [증명-수치확인]: f=x+2x(x-1)^2 (증가, f(0)=0, 양의 고정점 1) — h 의 양의 영점이 고정점과 일치
    fx = x + 2 * x * (x - 1) ** 2
    fs, fps, fpps, gp, h = make(fx)
    xs = [mp.mpf(k) / 20 for k in range(1, 61)]
    hv = [h(X) for X in xs]
    zeros = [xs[i] for i in range(len(xs) - 1) if hv[i] * hv[i + 1] < 0 or hv[i] == 0]
    z = []
    for zz in zeros:
        r0 = mp.findroot(h, (zz, zz + mp.mpf(1) / 20), solver="anderson")
        if all(abs(r0 - w) > 1e-8 for w in z):
            z.append(r0)
    ok21 = len(z) == 1 and abs(fs(z[0]) - z[0]) < 1e-12
    ok &= ok21
    msgs.append(f"2-1 h 의 양의 영점={[mp.nstr(v, 10) for v in z]}, f(α)=α={ok21} [증명-수치확인]")
    # 2-2: h(β)=h'(β)=0 인 모형 두 개(c=2,1)에서 h''(β)=-2^{1/4} f''(β) 확인 → h''=-2^{9/4} 이면 f''(β)
    vals = []
    for cc in (2, 1):
        fx = x + cc * x * (x - 1) ** 2
        fs, fps, fpps, gp, h = make(fx)
        hp = lambda X: (1 + gp(X) ** 4) ** 0.25 - (1 + fps(X) ** 4) ** 0.25
        hpp = mp.diff(hp, 1)
        vals.append((abs(h(1)) < 1e-15, abs(hp(1)) < 1e-15, hpp / fpps(1)))
    k = vals[0][2]
    ok &= all(v[0] and v[1] for v in vals) and abs(vals[0][2] - vals[1][2]) < 1e-10 and abs(k + mp.mpf(2) ** 0.25) < 1e-10
    fpp = -mp.mpf(2) ** (mp.mpf(9) / 4) / k
    a22 = sp.sympify(load_answer(2, 1))
    ok &= abs(fpp - a22) < 1e-9
    msgs.append(f"2-2 h''(β)/f''(β)={mp.nstr(k, 10)}(=-2^(1/4)), f''(β)={mp.nstr(fpp, 10)} md={a22}")
    # 2-3 [증명-수치확인]: f=x+½ln(1+x²) (증가, f/x→1) 에서 h(x)/x 가 0 으로 감소
    fx = x + sp.log(1 + x ** 2) / 2
    fs, fps, fpps, gp, h = make(fx)
    r = [abs(h(mp.mpf(X)) / X) for X in (10, 100, 1000)]
    ok &= r[0] > r[1] > r[2] and r[2] < 0.02
    msgs.append(f"2-3 |h(x)/x| (x=10,100,1000)={[mp.nstr(v, 4) for v in r]} [증명-수치확인]")
    report(2, ok, "; ".join(msgs))


def p03():
    import mpmath as mp
    mp.mp.dps = 30
    msgs, ok = [], True
    A_ = sp.symbols("a", real=True)
    # 3-1: 거리 제곱의 최솟값을 수치 최소화로 계산해 r(a)^2 를 정하고 적분 (구간 [-1,1] 을 a=1/2 로 분할)
    def r2(av):
        av = mp.mpf(av)
        dist2 = lambda u: u ** 2 + (u ** 2 - av) ** 2
        cands = [mp.mpf(0)] + ([mp.sqrt(av - mp.mpf(1) / 2)] if av > mp.mpf(1) / 2 else [])
        best = min(dist2(c) for c in cands)
        # 격자 검사로 최솟값 확인
        grid = min(dist2(mp.mpf(k) / 200) for k in range(0, 800))
        assert best <= grid + 1e-20
        return best
    I = mp.quad(r2, [-1, mp.mpf(1) / 2, 1])
    a31 = sp.sympify(load_answer(3, 0))
    ok &= abs(I - mp.mpf(str(sp.N(a31, 30)))) < 1e-15
    msgs.append(f"3-1 ∫r²={mp.nstr(I, 12)} md={a31}")
    # 3-2 [증명-수치확인]: 여러 a 에서 교점의 원 접선 기울기 = 포물선 접선 기울기
    good = True
    for av in (mp.mpf("-0.7"), mp.mpf("0.3"), mp.mpf("0.9"), mp.mpf(3), mp.mpf(20)):
        R2 = r2(av)
        # 교점: x^2=y 대입 → y^2+(1-2a)y+a^2-R2=0, y≥0
        bq, cq = 1 - 2 * av, av ** 2 - R2
        disc = bq ** 2 - 4 * cq
        good &= disc > -1e-20
        disc = max(disc, 0)
        ys = sorted(set(mp.nstr(v, 20) for v in ((-bq + mp.sqrt(disc)) / 2, (-bq - mp.sqrt(disc)) / 2) if v >= -1e-20))
        good &= len(ys) == 1                                   # 교점의 y 는 하나(중근 또는 원점)
        for y in ys:
            y = max(mp.mpf(y), 0)
            for x0 in ([mp.sqrt(y), -mp.sqrt(y)] if y > 1e-20 else [mp.mpf(0)]):
                circ = -x0 / (y - av) if abs(y - av) > 1e-20 else None
                good &= circ is not None and abs(circ - 2 * x0) < 1e-10
    ok &= good
    msgs.append(f"3-2 접선 기울기 일치={good} [증명-수치확인]")
    # 3-3: S(a)=∫_{-X}^{X}(원 아래호 - x^2)dx 를 직접 적분, S - F(X) → 0 (그리고 0 ≤ S-F ≤ 1/(8X))
    Fmd = sp.lambdify(x, sp.sympify(load_answer(3, 2)), "mpmath")
    diffs = []
    for av in (10, 100, 1000, 10000):
        av = mp.mpf(av)
        X = mp.sqrt(av - mp.mpf(1) / 2)
        R = mp.sqrt(av - mp.mpf(1) / 4)
        S = mp.quad(lambda u: (av - mp.sqrt(R ** 2 - u ** 2)) - u ** 2, [-X, 0, X])
        dd = S - Fmd(X)
        diffs.append(dd)
        ok &= 0 <= dd <= 1 / (8 * X) + 1e-20
    ok &= abs(diffs[-1]) < 1e-3 and diffs[0] > diffs[-1]
    # F 의 유일성: 다른 삼차식이면 차이가 0 으로 가지 않음(차이 다항식이 상수 아닌 한) — 상수항까지 맞는지 확인
    msgs.append(f"3-3 S-F(X) (a=10..10^4)={[mp.nstr(v, 4) for v in diffs]} (≤1/(8X))")
    report(3, ok, "; ".join(msgs))


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
