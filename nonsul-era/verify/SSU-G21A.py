#!/usr/bin/env python3
"""숭실대 2021학년도 논술 자연1 (SSU-G21A-01·02·04; 3번 확통 제외) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수(번호 = 원문 문제 번호)를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G21A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, locals_=None):
    """yaml 의 subparts[sub].answer.value 를 sympy 로 읽는다."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return sp.sympify(d["subparts"][sub]["answer"]["value"], locals=locals_ or {})


def close(a, b, tol=1e-9):
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(1.0, abs(b))


def p01():
    """위치 A(t)=∫(6s-2)ds, B(t)=A(t^2). (1) B=A 의 양수 해, (2) [0, 마지막 만남]에서 |B-A| 최대."""
    s = sp.symbols("s", real=True)
    A = sp.integrate(6 * s - 2, (s, 0, t))
    B = A.subs(t, t ** 2)
    d = sp.expand(B - A)
    meets = sorted(r for r in sp.solve(d, t) if r.is_real and r > 0)
    ans1 = load_answer(1, 0)
    ok1 = len(meets) == len(ans1) and all(sp.simplify(u - v) == 0 for u, v in zip(meets, sorted(ans1, key=float)))
    T = meets[-1]
    cands = [c for c in sp.solve(sp.diff(d, t), t) if c.is_real and 0 < c < T] + [sp.Integer(0), T]
    best = max(cands, key=lambda c: float(abs(d.subs(t, c))))
    dist = sp.nsimplify(sp.radsimp(abs(d.subs(t, best))))
    ans2 = load_answer(1, 1)
    ok2 = sp.simplify(best - ans2[0]) == 0 and sp.simplify(dist - ans2[1]) == 0
    # 대표 오답(MC-CAL-0711): 극댓값만 보면 (2*sqrt(3)-1)/12
    wrong = max(float(d.subs(t, c)) for c in cands)
    report(1, ok1 and ok2, f"(1) 만남={meets} (2) t*={best}, 거리={dist} [극댓값만 보면 {wrong:.6f}], md={ans1},{ans2}")


def p02():
    """조건을 만족하는 f 를 실제로 구성(φ(x)=c x^p)해 ∫_0^{f(1)} g 를 수치 계산하고 a 에 대해 최대화."""
    import mpmath as mp
    mp.mp.dps = 30

    def build(a):
        c = mp.mpf(3) / (a + 1)                 # (iii) x=0
        p = (2 * a - 4) / (a + 1)               # ∫_0^1 φ = c/(p+1) 를 (iv) 가 요구하는 값이 되게
        phi = lambda X: c * X ** p
        f = lambda X: phi(X) if X <= 1 else 3 - a * phi(2 - X)   # (iii): f(1+x)=3-a f(1-x)
        return c, p, f

    def G(a):
        c, p, f = build(a)
        total = mp.quad(f, [0, 1, 2])
        assert abs(total - 2) < 1e-20, total                      # (iv) 확인
        assert all(f(mp.mpf(i) / 50) < f(mp.mpf(i + 1) / 50) for i in range(100))   # 증가 확인
        ginv = lambda y: (y / c) ** (1 / p)                      # φ 의 역함수
        return mp.quad(ginv, [0, f(1)])

    a0 = mp.findroot(lambda a: mp.diff(G, a), mp.mpf(3.7))
    ans = load_answer(2)
    ok = abs(a0 - mp.mpf(str(sp.N(ans, 40)))) < 1e-15 and G(a0) > G(a0 - mp.mpf("0.01")) and G(a0) > G(a0 + mp.mpf("0.01"))
    report(2, ok, f"argmax a={mp.nstr(a0, 15)}, 최댓값={mp.nstr(G(a0), 12)}, md={ans}")


def p04():
    """교점 P 좌표로 ∠PRQ 를 직접 계산, A(t)=½(t²-1)∠PRQ 의 도함수를 큰 t 에서 고정밀 수치로."""
    import mpmath as mp
    mp.mp.dps = 60

    def A(T):
        xp = 1 / T                                # x^2+y^2=1, (x-T)^2+y^2=T^2-1 의 교점
        yp = mp.sqrt(1 - xp ** 2)
        ang = 2 * mp.atan2(yp, T - xp)
        return (T ** 2 - 1) * ang / 2

    vals = [mp.diff(A, mp.mpf(10) ** e) for e in (4, 6, 8)]
    ans = load_answer(4)
    ok = abs(vals[-1] - ans) < 1e-12 and abs(vals[0] - ans) < 1e-6
    report(4, ok, f"dA/dt at t=1e4,1e6,1e8: {[mp.nstr(v, 12) for v in vals]}, md={ans}")


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
