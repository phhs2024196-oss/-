#!/usr/bin/env python3
"""경북대 2027학년도 논술(AAT) 모의고사 자연계열Ⅱ (KNU-M27B-01~02) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import random
import re
import sys

import sympy as sp
import mpmath as mp

PREFIX = "KNU-M27B"
WIDTH = 2
RESULTS = []
x = sp.Symbol("x", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return None if v is None else sp.sympify(v)


def p01():
    mp.mp.dps = 20
    A_, B_ = sp.symbols("a b", positive=True)
    # 1-1: f' = (ax^2+(2a+b)x+b)e^x, 계수가 모두 양수 → x>0 에서 양수; f(0)=0, f→∞
    fp = sp.simplify(sp.diff((A_ * x ** 2 + B_ * x) * sp.exp(x), x) * sp.exp(-x))
    coeffs = sp.Poly(sp.expand(fp), x).all_coeffs()
    ok1 = all(sp.ask(sp.Q.positive(cf), sp.Q.positive(A_) & sp.Q.positive(B_)) for cf in coeffs)

    def make(a, b):
        f = lambda u: (a * u ** 2 + b * u) * mp.e ** u
        def finv(y):
            if y == 0:
                return mp.mpf(0)
            hi = mp.mpf(1)
            while f(hi) < y:
                hi *= 2
            return mp.findroot(lambda u: f(u) - y, (mp.mpf(0), hi), solver="anderson")
        return f, finv

    # 1-2: g'(1)=2, g(1)=5/3 을 역함수 수치·수치적분으로 직접 풀기
    def eqs(a, b):
        f, finv = make(a, b)
        e1 = 2 - finv(1) + f(1) - 2
        e2 = mp.quad(lambda u: 2 - finv(u) + f(u), [0, 1]) - mp.mpf(5) / 3
        return [e1, e2]
    sol = mp.findroot(eqs, (mp.mpf("0.12"), mp.mpf("0.25")))
    a0, b0 = sol[0], sol[1]
    ans = load_answer(1, 1)
    ok2 = abs(a0 - float(ans[0])) < 1e-12 and abs(b0 - float(ans[1])) < 1e-12
    # 1-3: [0,1] 에서 finv - f >= 0 [증명-수치확인]
    f, finv = make(a0, b0)
    ok3 = all(finv(mp.mpf(k) / 400) - f(mp.mpf(k) / 400) >= -mp.mpf("1e-15") for k in range(0, 401))
    # 1-4: g(x)=1/n 의 근 d_n, n d_n 의 수렴
    g = lambda X: mp.quad(lambda u: 2 - finv(u) + f(u), [0, X])
    vals = []
    for n in (10, 100, 1000, 10000):
        dn = mp.findroot(lambda X: g(X) - mp.mpf(1) / n, (mp.mpf(0), mp.mpf(1)), solver="anderson")
        vals.append(n * dn)
    target = load_answer(1, 3)
    errs = [abs(v - float(target)) for v in vals]
    ok4 = errs[-1] < 1e-3 and all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    report(1, ok1 and ok2 and ok3 and ok4,
           f"f'계수 양수={ok1}, (a,b)≈({mp.nstr(a0,12)},{mp.nstr(b0,12)}) md={ans}, finv-f>=0={ok3}, "
           f"n d_n={[mp.nstr(v,8) for v in vals]} → md={target} [증명-수치확인]")


def p02():
    mp.mp.dps = 30
    a = 2 + mp.sqrt(3)
    A = (1 - a, 1 - 2 * a)
    B = (1 + a, 1 + 2 * a)

    def hits(E, R):
        """선분 E-R(끝점 제외)과 y=x^2 의 교점 매개변수 u∈(0,1) 목록."""
        dx, dy = R[0] - E[0], R[1] - E[1]
        c2, c1, c0 = -dx ** 2, dy - 2 * E[0] * dx, E[1] - E[0] ** 2
        if abs(c2) < mp.mpf("1e-25"):
            us = [] if abs(c1) < mp.mpf("1e-25") else [-c0 / c1]
        else:
            D = c1 ** 2 - 4 * c2 * c0
            if D < 0:
                return []
            us = [(-c1 + sg * mp.sqrt(D)) / (2 * c2) for sg in (1, -1)]
            if D == 0:
                us = us[:1]
        return sorted(set(u for u in us if 0 < u < 1))

    def PQ(R):
        if abs(R[1] - R[0] ** 2) < mp.mpf("1e-20"):
            return None
        hA, hB = hits(A, R), hits(B, R)
        if len(hA) != 1 or len(hB) != 1:
            return None
        p = A[0] + hA[0] * (R[0] - A[0])
        q = B[0] + hB[0] * (R[0] - B[0])
        if not (p < 1 < q):
            return None
        return p, q

    # 2-3 [증명-수치확인]: (Ⅰ)~(Ⅳ) 만족하는 무작위 R 은 모두 x1^2<y1, 2-2: f(r)=0 양의 근 유일
    random.seed(7)
    nvalid, bad, bad22 = 0, 0, 0
    for _ in range(40000):
        R = (mp.mpf(random.uniform(-12, 12)), mp.mpf(random.uniform(-40, 80)))
        r = PQ(R)
        if r is None:
            continue
        nvalid += 1
        if not R[0] ** 2 < R[1]:
            bad += 1
        x1, y1 = R
        for (s0, s1, w) in ((1 - a, 1 - 2 * a, None), (1 + a, 1 + 2 * a, None)):
            # f(r)=(s0+r x1)^2-(1+r)(s1+r y1) 의 양의 근 개수
            c2, c1, c0 = x1 ** 2 - y1, 2 * s0 * x1 - s1 - y1, s0 ** 2 - s1
            D = c1 ** 2 - 4 * c2 * c0
            roots = [] if D < 0 else [(-c1 + sg * mp.sqrt(D)) / (2 * c2) for sg in (1, -1)]
            if sum(1 for z in set(roots) if z > 0) != 1:
                bad22 += 1
    ok23 = nvalid > 100 and bad == 0 and bad22 == 0

    # 넓이: 폐곡선 A→P(선분)→M(포물선 호)→A(선분), Green 정리
    def area_APM(E, P, Mx=1):
        def seg(U, V):
            return (U[0] * V[1] - V[0] * U[1]) / 2
        Pp = (P, P ** 2)
        arc = (mp.mpf(Mx) ** 3 - P ** 3) / 6          # ∫_P^1 x^2/2 dx  (x dy - y dx)/2 = x^2/2 dx
        return abs(seg(E, Pp) + arc + seg((1, 1), E))

    def areas(R):
        r = PQ(R)
        if r is None:
            return None
        p, q = r
        tri = abs((B[0] - A[0]) * (R[1] - A[1]) - (R[0] - A[0]) * (B[1] - A[1])) / 2
        return p, q, area_APM(A, p), area_APM(B, q), tri

    # 2-4(전반) S_APM=S_BQM 이면 p+q=2: y1 여러 값에서 x1 을 풀어 확인
    okpq = True
    cnt = 0
    for y1 in [mp.mpf(v) for v in (3, 5, 8, 12, 20)]:
        for x0 in (mp.mpf("0.5"), mp.mpf(1), mp.mpf("1.5")):
            try:
                x1 = mp.findroot(lambda X: areas((X, y1))[2] - areas((X, y1))[3], x0)
            except Exception:  # noqa: BLE001
                continue
            r = areas((x1, y1))
            if r is None:
                continue
            cnt += 1
            okpq &= abs(r[0] + r[1] - 2) < 1e-15
    # 2-4(후반) S_APM=S_BQM=S_ABR/3 을 (x1,y1) 에 대해 직접 풀기
    sols = set()
    for x0 in (0.6, 1.0, 1.4):
        for y0 in (1.5, 2.5, 4, 8):
            try:
                def F(X, Y):
                    r = areas((X, Y))
                    return [r[2] - r[3], r[2] - r[4] / 3]
                s = mp.findroot(F, (mp.mpf(x0), mp.mpf(y0)))
                if areas((s[0], s[1])) is not None:
                    sols.add((mp.nstr(s[0], 18), mp.nstr(s[1], 18)))
            except Exception:  # noqa: BLE001
                pass
    ans = load_answer(2, 3)
    ok24 = len(sols) == 1 and all(abs(mp.mpf(u) - float(ans[0])) < 1e-15 and abs(mp.mpf(v) - float(ans[1])) < 1e-15
                                  for u, v in sols)
    report(2, ok23 and okpq and cnt > 3 and ok24,
           f"(Ⅰ)~(Ⅳ) 만족 표본 {nvalid}개 중 x1^2>=y1 {bad}개, 양근 유일 위반 {bad22}개; "
           f"S_APM=S_BQM ⇒ p+q=2 ({cnt}건)={okpq}; R 해={sols}, md={ans} [증명-수치확인]")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda q: int(q[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
