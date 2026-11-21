#!/usr/bin/env python3
"""숭실대 2022학년도 논술고사(자연1) (SSU-G22A-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
yaml 의 answer.value(sympy 문법, 'a=.., b=..' 꼴은 이름별 dict, 'L=R' 꼴은 L-R 식)를 읽어 독립 재계산 값과 비교한다.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SSU-G22A"
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
    """조건 (i)(ii)를 만족하는 서로 다른 g 두 개(g=x^k, g=x^k(1+ε sin(2π log2 x)), k=log2 3)로 직접 계산해
    (1) f'(3)g'(1), (2) ∫0^1 g / ∫1^2 g 가 g 에 무관하게 md 값과 같은지 확인."""
    k = mp.log(3) / mp.log(2)
    gs = [lambda X: X**k,
          lambda X: X**k * (1 + mp.mpf("0.05") * mp.sin(2 * mp.pi * mp.log(X) / mp.log(2))) if X > 0 else mp.mpf(0)]
    ans1, ans2 = load_answer(1, "1"), load_answer(1, "2")
    ok, msg = True, []
    for g in gs:
        assert abs(g(mp.mpf(1)) - 1) < 1e-25 and abs(g(mp.mpf("1.7") * 2) - 3 * g(mp.mpf("1.7"))) < 1e-20
        # g 가 증가함수(역함수 존재) 확인
        assert all(mp.diff(g, mp.mpf(i) / 50) > 0 for i in range(1, 200))
        g1 = mp.diff(g, 1)
        x3 = mp.findroot(lambda X: g(X) - 3, 2)             # g(x3)=3
        fprime3 = 1 / mp.diff(g, x3)                         # 역함수의 미분
        v1 = fprime3 * g1
        A = mp.quad(g, [1, 2])
        B = mp.quad(g, [0, 0.5, 1])
        v2 = sp.N(ans2.subs(sp.Symbol("A"), sp.Float(str(A), 30)), 25)
        ok = ok and abs(v1 - mp.mpf(str(sp.N(ans1, 30)))) < 1e-15 and abs(B - mp.mpf(str(v2))) < 1e-12
        msg.append(f"f'(3)g'(1)={mp.nstr(v1, 12)}, B/A={mp.nstr(B / A, 12)}")
    report(1, ok, "; ".join(msg) + f", md=({ans1}, {ans2})")


def p02():
    """p=f-g 는 최고차 계수 1 인 삼차식, 서로 다른 실근 정확히 2개(큰 근 2), h 연속 ⇒ lim h = 16.
    두 꼴을 모두 풀어 조건을 sympy limit 로 다시 확인한 뒤 넓이를 적분."""
    a = sp.symbols("a", real=True)
    found = []
    for p in ((x - a) * (x - 2) ** 2, (x - a) ** 2 * (x - 2)):
        for av in sp.solve(sp.diff(p, x).subs(x, 2) - 16, a):
            if av.is_real and av < 2:
                pp = sp.expand(p.subs(a, av))
                L = sp.limit((sp.exp(pp) - 1) / (x - 2), x, 2)
                if L == 16 and len(set(sp.real_roots(sp.Poly(pp, x)))) == 2:
                    found.append(pp)
    areas = set()
    for pp in found:
        rts = sorted(set(sp.real_roots(sp.Poly(pp, x))))
        # 두 교점 사이(부호 일정)에서 |f-g| 적분
        areas.add(sp.Abs(sp.integrate(pp, (x, rts[0], rts[-1]))))
    ans = load_answer(2)
    ok = len(found) == 1 and areas == {ans}
    report(2, ok, f"p(x)={found}, 넓이={areas}, md={ans}")


def p03():
    """R=7√2/4, cosA=5/7, 넓이 √6 으로 a,b,c 를 sympy 로 직접 풀어 md 집합과 비교."""
    a, b, c = sp.symbols("a b c", positive=True)
    R = 7 * sp.sqrt(2) / 4
    cA = sp.Rational(5, 7)
    sA = sp.sqrt(1 - cA**2)
    sols = sp.solve([a - 2 * R * sA, b * c * sA / 2 - sp.sqrt(6), a**2 - (b**2 + c**2 - 2 * b * c * cA)], [a, b, c], dict=True)
    sides = {tuple(sorted([sp.nsimplify(sp.simplify(s[v])) for v in (a, b, c)], key=lambda z: float(z))) for s in sols}
    ans = load_answer(3)
    md = tuple(sorted([sp.nsimplify(z) for z in ans], key=lambda z: float(z)))
    # 해가 (b,c) 교환만 다른 것이므로 세 변의 집합은 하나
    ok = len(sides) == 1 and all(sp.simplify(p - q) == 0 for p, q in zip(next(iter(sides)), md))
    report(3, ok, f"세 변={sides}, md={ans}")


def p04():
    """x_n 으로부터 b_n(구간 안 근 2개 ⇒ 길이 = 3π/b_n), 각 x_n 에서의 미분가능성으로 a_n 을 차례로 정해
    f 를 구간별로 만들고 ∫_{x_1}^{x_10} f 를 sympy 로 직접 적분."""
    X = lambda k: 3 * sp.pi * (1 - sp.Rational(1, 2 ** (k - 1)))
    bs, as_ = {}, {}
    for k in range(1, 11):
        bs[k] = sp.simplify(3 * sp.pi / (X(k + 1) - X(k)))
        # 끝점이 sin 의 영점인지(조건 (ii)) 확인
        assert sp.simplify(sp.sin(bs[k] * X(k))) == 0 and sp.simplify(sp.sin(bs[k] * X(k + 1))) == 0
    as_[1] = sp.Rational(1, 2) / (bs[1] * sp.cos(bs[1] * X(1)))   # f'(x_1)=1/2 (오른쪽 미분계수)
    for k in range(2, 11):
        left = as_[k - 1] * bs[k - 1] * sp.cos(bs[k - 1] * X(k))
        as_[k] = sp.simplify(left / (bs[k] * sp.cos(bs[k] * X(k))))
    I = sum(sp.integrate(as_[k] * sp.sin(bs[k] * x), (x, X(k), X(k + 1))) for k in range(1, 10))
    I = sp.nsimplify(sp.simplify(I))
    ans1, ans2 = load_answer(4, "1"), load_answer(4, "2")
    ok = bs[10] == ans1 and sp.simplify(I - ans2) == 0
    report(4, ok, f"b_10={bs[10]}, a_n={[as_[k] for k in range(1, 5)]}…, 적분={I}, md=({ans1}, {ans2})")


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
