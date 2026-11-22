#!/usr/bin/env python3
"""한국외대 2023학년도 논술전형 모의논술(자연계) 문제 1~10 (HUFS-M23-01~10) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "HUFS-M23"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, c = sp.symbols("a b c", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "c": c}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/HUFS/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """각 n 에 대해 r^n = -n^2+7n-10 의 실근을 sympy 로 구하고 음의 실근 존재 여부를 직접 판정."""
    r = sp.symbols("r", real=True)
    good = []
    for nn in range(2, 11):
        base = -nn**2 + 7 * nn - 10
        roots = sp.solve(sp.Eq(r**nn, base), r)
        if any(rt.is_real and rt < 0 for rt in roots):
            good.append(nn)
    ans = load_answer(1)
    report(1, sum(good) == ans, f"음의 n제곱근 존재 n={good}, 합={sum(good)}, yaml={ans}")


def p02():
    """[0,4] 에서 log_{1/2}(x^2-2x+8) 의 최솟값: 임계점·끝점 후보에서 sympy 로 직접 비교."""
    y = sp.log(x**2 - 2 * x + 8, sp.Rational(1, 2))
    cands = [sp.Integer(0), sp.Integer(4)] + [r for r in sp.solve(sp.diff(y, x), x) if 0 <= r <= 4]
    m = min((sp.nsimplify(sp.simplify(y.subs(x, cc))) for cc in cands), key=lambda v: float(v))
    ans = load_answer(2)
    report(2, sp.simplify(m - ans) == 0, f"후보 x={cands}, 최솟값={m}, yaml={ans}")


def p03():
    """a1 을 기호로 두고 (가)(나)로 a_1..a_20 을 실제로 생성, (다) 방정식을 풀어 비교."""
    a1 = sp.symbols("a1")
    seq = [a1 * (-2) ** k for k in range(5)]
    for m in range(1, 16):                      # a_{m+5} = a_m + m + 1
        seq.append(seq[m - 1] + m + 1)
    sol = sp.solve(sp.Eq(sum(seq), 132), a1)
    ans = load_answer(3)
    report(3, len(sol) == 1 and sol[0] == ans, f"Σ a_n = {sp.expand(sum(seq))}, a1={sol}, yaml={ans}")


def p04():
    """c>0 에서 min_x f = f(3c) 를 임계점 전체로 계산 → min f = 0 이 되는 c 를 풀고, 그 c 의 앞뒤 격자에서 부호 확인."""
    f = x**4 - 4 * c * x**3 + 3
    def minval(cv):
        g = f.subs(c, cv)
        crit = [r for r in sp.solve(sp.diff(g, x), x) if r.is_real]
        return min(sp.nsimplify(g.subs(x, r)) for r in crit)
    cp = sp.symbols("cp", positive=True)
    gm = sp.expand(f.subs(c, cp).subs(x, 3 * cp))       # c>0 일 때 최솟값 후보 f(3c)
    cmax = max(sp.solve(sp.Eq(gm, 0), cp))
    ans = load_answer(4)
    eps = sp.Rational(1, 1000)
    ok = sp.simplify(cmax - ans) == 0 and minval(ans) >= 0 and minval(ans + eps) < 0 \
        and all(minval(sp.Rational(k, 20)) >= 0 for k in range(-11, 12))   # |c|<=0.55 격자에서 f>=0
    report(4, ok, f"f(3c)={gm}, 경계 c={cmax}, min f(c=ans)={minval(ans)}, c=ans+0.001 에서 음수, yaml={ans}")


def p05():
    """격자 정사각형 [i,i+1]x[j,j+1] (1<=i<=99, 1<=j<=5) 전수: 곡선이 x∈[i,i+1] 에서 갖는 값의 구간
    [log k i, log k(i+1)] 이 [j, j+1] 과 겹치는지 정확한 정수 비교(10 의 거듭제곱)로 판정."""
    def meets(k, i, j):          # log10(k*i) <= j+1  and  log10(k*(i+1)) >= j
        return k * i <= 10 ** (j + 1) and k * (i + 1) >= 10 ** j
    cnt = sum(1 for i in range(1, 100) for j in range(1, 6) if meets(3, i, j) and meets(8, i, j))
    # 교차 판정 자체를 부동소수 표본으로도 확인(곡선 위 표본점이 정사각형 안에 있는지)
    import math
    def meets_num(k, i, j):
        return any(j - 1e-12 <= math.log10(k * (i + s / 400)) <= j + 1 + 1e-12 for s in range(401))
    cnt2 = sum(1 for i in range(1, 100) for j in range(1, 6) if meets_num(3, i, j) and meets_num(8, i, j))
    ans = load_answer(5)
    report(5, cnt == ans == cnt2, f"정수 판정 {cnt}개, 표본 판정 {cnt2}개, yaml={ans}")


def p06():
    """R=2√3 로 a=2R sinA 등. 각 A 를 몇 개 고르고 (나) 를 B 에 대해 수치로 풀어(a≠b 인 해) c 가 항상 같은지 확인 + 기호 유도."""
    import random
    import mpmath as mp
    R = 2 * sp.sqrt(3)
    A_, B_ = sp.symbols("A B", positive=True)
    C_ = sp.pi - A_ - B_
    av, bv, cv = 2 * R * sp.sin(A_), 2 * R * sp.sin(B_), 2 * R * sp.sin(C_)
    expr = (av - bv) * sp.sin(C_)**2 - av * sp.sin(A_)**2 + bv * sp.sin(B_)**2
    random.seed(1)
    cs = []
    for _ in range(8):
        A0 = random.uniform(0.05, 0.95)
        g = sp.lambdify(B_, expr.subs(A_, A0), "mpmath")
        for k in range(1, 40):
            try:
                r = mp.findroot(g, k * (mp.pi - A0) / 40)
            except Exception:  # noqa: BLE001
                continue
            if 0 < r < mp.pi - A0 and abs(mp.sin(A0) - mp.sin(r)) > 1e-6:   # a ≠ b
                cs.append(float(cv.subs({A_: A0, B_: r})))
                break
    a_, b_ = sp.symbols("a_ b_", positive=True); cc = sp.symbols("cc", real=True)
    cosC = sp.solve(sp.Eq(a_**2 + a_ * b_ + b_**2, a_**2 + b_**2 - 2 * a_ * b_ * cc), cc)[0]
    csym = sp.simplify(2 * R * sp.sin(sp.acos(cosC)))
    ans = load_answer(6)
    ok = len(cs) >= 5 and all(abs(v - float(ans)) < 1e-9 for v in cs) and csym == ans
    report(6, ok, f"cosC={cosC}, c=2R sinC={csym}, 수치해 {len(cs)}개 모두 c≈{[round(v, 9) for v in cs[:3]]}, yaml={ans}")


def p07():
    """일반 삼차 f=4x^3+px^2+qx+r ((나) 로 최고차 계수 4·차수 3) 에 대해 (가) 극한이 10 이 되는 계수를 직접 구한다:
    x f(x) 를 (x-2) 거듭제곱으로 전개해 0차·1차 계수 = 0, 2차 계수 = 10."""
    p, q, r = sp.symbols("p q r")
    f = 4 * x**3 + p * x**2 + q * x + r
    assert sp.limit(f / (2 * x**3 - 3), x, sp.oo) == 2
    u = sp.symbols("u")
    g = sp.expand((x * f).subs(x, u + 2))
    eqs = [sp.Eq(g.coeff(u, 0), 0), sp.Eq(g.coeff(u, 1), 0), sp.Eq(g.coeff(u, 2), 10)]
    sol = sp.solve(eqs, [p, q, r], dict=True)
    fs = f.subs(sol[0])
    lim = sp.limit(x * fs / (x - 2)**2, x, 2)
    ans = load_answer(7)
    report(7, len(sol) == 1 and lim == 10 and fs.subs(x, 1) == ans,
           f"f={sp.factor(fs)}, 극한={lim}, f(1)={fs.subs(x, 1)}, yaml={ans}")


def p08():
    """P=(2cos30°,2sin30°) 로 a 를 구하고, 색칠 영역(원 내부·포물선 위)을 ∫(√(4-x²) - a x²)dx 로 직접 적분해 b, c 를 분리."""
    P = (2 * sp.cos(sp.pi / 6), 2 * sp.sin(sp.pi / 6))
    av = sp.solve(sp.Eq(a * P[0]**2, P[1]), a)[0]
    S = sp.simplify(sp.integrate(sp.sqrt(4 - x**2) - av * x**2, (x, -P[0], P[0])))
    bb = sp.simplify(S.coeff(sp.pi))
    cc = sp.simplify((S - bb * sp.pi) / sp.sqrt(3))
    ans = load_answer(8)
    ok = bb.is_rational and cc.is_rational and sp.simplify(av * bb * cc - ans) == 0
    report(8, ok, f"a={av}, 넓이={S}, b={bb}, c={cc}, abc={av * bb * cc}, yaml={ans}")


def p09():
    """[증명-수치확인] (1) 귀납 단계 항등식: 3(a^{k+1}+2b^{k+1}) - (a^k+2b^k)(a+2b) = 2(b^k-a^k)(b-a) 를 k=1..30 기호 확인.
    (2) 원 부등식을 0<=a<=b 격자(유리수 21x21, b<=5) × n=1..40 에서 정확 비교. answer 는 proof(값 없음)."""
    ok_id = all(sp.expand(3 * (a**(k + 1) + 2 * b**(k + 1)) - (a**k + 2 * b**k) * (a + 2 * b)
                          - 2 * (b**k - a**k) * (b - a)) == 0 for k in range(1, 31))
    viol = 0
    grid = [sp.Rational(i, 4) for i in range(0, 21)]
    for av in grid:
        for bv in grid:
            if av > bv:
                continue
            for nn in range(1, 41):
                if ((av + 2 * bv) / 3)**nn > (av**nn + 2 * bv**nn) / 3:
                    viol += 1
    kind = load_answer(9, raw=True)
    report(9, ok_id and viol == 0 and kind is None,
           f"[증명-수치확인] 귀납 항등식 k=1..30 {ok_id}, 격자·n=1..40 위반 {viol}건, answer.value={kind}")


def p10():
    """(가)(나) 를 만족하는 구체 함수 f=3+K ln x 를 기호로 찾아(조건을 sympy 로 확인) f'(3) 계산.
    또 미분계수 정의식 lim (f(1+h/3)-3)/h 를 직접 극한으로 계산해 일치 확인."""
    K = sp.symbols("K")
    p_, q_ = sp.symbols("p q", positive=True)
    f = lambda X: 3 + K * sp.log(X)
    assert sp.simplify(sp.expand_log(f(p_ * q_) - (f(p_) + f(q_) - 3), force=True)) == 0   # (가)
    Kv = sp.solve(sp.Eq(sp.diff(f(x), x).subs(x, 1), 3), K)[0]                              # (나)
    fx = f(x).subs(K, Kv)
    d3 = sp.diff(fx, x).subs(x, 3)
    h = sp.symbols("h")
    lim = sp.limit((fx.subs(x, 1 + h / 3) - 3) / h, h, 0)
    ans = load_answer(10)
    report(10, d3 == ans and lim == ans, f"f=3+{Kv}ln x 가 (가)(나) 만족, f'(3)={d3}, 정의식 극한={lim}, yaml={ans}")


# ── 실행 ──
if __name__ == "__main__":
    for name in sorted((k for k in dict(globals()) if re.fullmatch(r"p\d+", k)), key=lambda s: int(s[1:])):
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            RESULTS.append(False)
            print(f"{PREFIX}-{int(name[1:]):0{WIDTH}d} FAIL  예외: {e!r}")
    print(f"SUMMARY M23: {sum(RESULTS)}/{len(RESULTS)} PASS")
    sys.exit(0 if all(RESULTS) else 1)
