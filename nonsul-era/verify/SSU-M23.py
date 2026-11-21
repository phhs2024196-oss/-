#!/usr/bin/env python3
"""숭실대 2023학년도 모의논술고사(자연) (SSU-M23-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
yaml 의 answer.value(sympy 문법, 'a=.., b=..' 꼴은 이름별 dict, 'L=R' 꼴은 L-R 식)를 읽어 독립 재계산 값과 비교한다.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SSU-M23"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_answer(s):
    """'1/4' → 식, '[..]' → 리스트, 'a=1, b=2' → dict, 'L=R' → L-R."""
    s = str(s).strip()
    loc = {"E": sp.E, "pi": sp.pi}
    if s.startswith("["):
        return list(sp.sympify(s, locals=loc))
    parts = _split_top(s)
    if all(re.fullmatch(r"[A-Za-z]\w*\s*=[^=].*", p) for p in parts):
        if len(parts) > 1 or re.fullmatch(r"[A-Za-z]\w*", parts[0].split("=")[0].strip()):
            return {p.split("=", 1)[0].strip(): sp.sympify(p.split("=", 1)[1], locals=loc) for p in parts}
    if "=" in s:
        l, r = s.split("=", 1)
        return sp.sympify(l, locals=loc) - sp.sympify(r, locals=loc)
    return sp.sympify(s, locals=loc)


def load_answer(seq, sub="1"):
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    sp_ = next(s for s in d["subparts"] if str(s["id"]) == str(sub))
    return parse_answer(sp_["answer"]["value"])


def close(a, b, tol=1e-12):
    return abs(complex(sp.N(a, 40)) - complex(sp.N(b, 40))) < tol


def p01():
    """원 C(중심 (1,R), R=tan(θ/2)) 와 외접하고 두 반직선에 접하는 작은 원을 수치로 구성, S(θ)/θ^2 를 θ→0 에서 평가."""
    def ratio(th):
        R = mp.tan(th / 2)
        # 작은 원 중심 (r/R, r): 중심거리 = R + r, r < R 인 근
        F = lambda r: (1 - r / R) ** 2 + (R - r) ** 2 - (R + r) ** 2
        r = mp.findroot(F, (R * mp.mpf("1e-6"), R * (1 - mp.mpf("1e-12"))), solver="bisect")
        D, A, Dp = (1, R), (1, 0), (r / R, r)
        S = abs((A[0] - Dp[0]) * (D[1] - Dp[1]) - (D[0] - Dp[0]) * (A[1] - Dp[1])) / 2
        return S / th ** 2
    vals = [ratio(mp.mpf(10) ** (-k)) for k in (4, 6, 8)]
    ans = load_answer(1)
    errs = [abs(v - mp.mpf(str(sp.N(ans, 30)))) for v in vals]
    ok = errs[-1] < 1e-8 and errs[0] > 50 * errs[1] > 2500 * errs[2]   # 오차가 θ 에 비례해 0 으로 감
    report(1, ok, f"S/θ² (θ=1e-4,1e-6,1e-8)={[mp.nstr(v, 12) for v in vals]}, md={ans}")


def p02():
    """(1) f(2)=3, f'(2)=4, f'(5)=0 의 1-모수 가족에서 '그래프와 x축의 교점 2개, x=5 극소' 를 만족하는 a 를 탐색.
    (2) 법선의 x절편 X(t)=t+f(t)f'(t) 의 (0,8/3) 최솟값."""
    a, b, c, d = sp.symbols("a b c d", real=True)
    F = a * x**3 + b * x**2 + c * x + d
    fam = sp.solve([F.subs(x, 2) - 3, sp.diff(F, x).subs(x, 2) - 4, sp.diff(F, x).subs(x, 5)], [b, c, d], dict=True)[0]
    Fa = F.subs(fam)

    def zeros_ok(av):
        G = sp.expand(Fa.subs(a, av))
        rts = {r for r in sp.real_roots(sp.Poly(G, x)) if r >= 2}
        minimum = sp.diff(G, x, 2).subs(x, 5) > 0
        return len(rts) == 1 and minimum   # (0,2) 쪽 근은 x=1 하나뿐
    cand = [s for s in sp.solve(Fa.subs(x, 5), a) if s != 0]
    good = [s for s in cand if zeros_ok(s)]
    # 다른 a 값에서는 조건이 깨지는지 확인
    others = all(not zeros_ok(sp.Rational(k, 7)) for k in range(-20, 21) if k != 0 and sp.Rational(k, 7) not in good)
    sol = {a: good[0], **{k: v.subs(a, good[0]) for k, v in fam.items()}} if len(good) == 1 else {}
    ans1 = load_answer(2, "1")
    ok1 = bool(sol) and others and all(sp.simplify(sol[sp.Symbol(k, real=True)] - v) == 0 for k, v in ans1.items()) and len(ans1) == 4
    f_left, f_right = x**2 - 1, Fa.subs(a, good[0]) if good else x
    X = lambda fx: x + fx * sp.diff(fx, x)
    crit = [r for r in sp.solve(sp.diff(X(f_left), x), x) if r.is_real and 0 < r < 2]
    cands = [X(f_left).subs(x, r) for r in crit]
    grid_min = min(min(float(X(f_left).subs(x, k / 1000)) for k in range(1, 2000)),
                   min(float(X(f_right).subs(x, 2 + k / 1500)) for k in range(0, 1000)))
    mn = min(cands)
    ans2 = load_answer(2, "2")
    ok2 = sp.simplify(mn - ans2) == 0 and grid_min >= float(mn) - 1e-9
    report(2, ok1 and ok2, f"(1) {sol} (2) 최솟값={sp.nsimplify(mn)} 격자최소={grid_min:.9f}, md=({ans1}, {ans2})")


def p03():
    """P(a,b) 에서 y=x^2 에 그은 두 접선의 ∠APB 를 직접 계산해 π/3 이 되는 a 를 수치로 찾고,
    md 의 관계식·넓이식에 대입해 확인."""
    def angle(a_, b_):
        r = mp.sqrt(a_**2 - b_)
        al, be = a_ - r, a_ + r
        PA = (al - a_, al**2 - b_)
        PB = (be - a_, be**2 - b_)
        cosv = (PA[0] * PB[0] + PA[1] * PB[1]) / (mp.sqrt(PA[0]**2 + PA[1]**2) * mp.sqrt(PB[0]**2 + PB[1]**2))
        return mp.acos(cosv), al, be
    rel = load_answer(3, "1")
    area = load_answer(3, "2")
    ok, msgs = True, []
    for bv in (mp.mpf(-1), mp.mpf("-0.8"), mp.mpf(-2), mp.mpf(-5)):
        g = lambda av: angle(av, bv)[0] - mp.pi / 3
        # a>=0 쪽 해(대칭). a=0 에서 각이 π/3 보다 작은지 확인 후 이분법
        if g(0) > 0:
            ok = False
            msgs.append(f"b={bv}: a=0 에서도 각>π/3")
            continue
        av = mp.findroot(g, (mp.mpf(0), mp.mpf(50)), solver="bisect")
        _, al, be = angle(av, bv)
        S = mp.quad(lambda X: X**2 - (2 * al * (X - al) + al**2), [al, av]) + \
            mp.quad(lambda X: X**2 - (2 * be * (X - be) + be**2), [av, be])
        r1 = sp.N(rel.subs({sp.Symbol("a"): sp.Float(str(av), 30), sp.Symbol("b"): sp.Float(str(bv), 30)}), 25)
        s_md = sp.N(area.subs(sp.Symbol("b"), sp.Float(str(bv), 30)), 25)
        good = abs(float(r1)) < 1e-15 and abs(float(s_md) - float(S)) < 1e-15
        ok = ok and good
        msgs.append(f"b={mp.nstr(bv,3)}: a={mp.nstr(av,10)} 관계식잔차={float(r1):.1e} S={mp.nstr(S,10)}")
    # b>-1/4 에서는 예각 π/3 이 될 수 없음(∠APB 가 둔각) 확인: b=-0.1 에서 각은 항상 > π/2
    obt = all(angle(mp.mpf(k) / 3, mp.mpf("-0.1"))[0] > mp.pi / 2 for k in range(0, 30))
    report(3, ok and obt, "; ".join(msgs) + f"; b=-0.1 둔각확인={obt}")


def p04():
    """세 펌프 중 급수·배수 순서쌍 6가지를 전수 조사, 0<=t<=1 에서 0<=물의 양<=5 인지 sympy 로 판정."""
    import itertools
    rates = {1: 8 * t + 1, 2: 4 * t + 2, 3: 6 * t + 4}
    s = sp.symbols("s", real=True)
    cnt, good = 0, []
    for i, j in itertools.permutations(rates, 2):
        W = sp.Rational(1, 10) + sp.integrate(rates[i] - rates[j], (t, 0, s))
        pts = [sp.Integer(0), sp.Integer(1)] + [r for r in sp.solve(sp.diff(W, s), s) if 0 < r < 1]
        vals = [W.subs(s, p) for p in pts]
        if min(vals) >= 0 and max(vals) <= 5:
            cnt += 1
            good.append((i, j))
    ans = load_answer(4)
    report(4, cnt == ans, f"가능한 (급수,배수)={good}, 개수={cnt}, md={ans}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
