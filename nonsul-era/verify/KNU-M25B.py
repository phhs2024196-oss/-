#!/usr/bin/env python3
"""KNU-M25B (경북대 2025 논술(AAT) 모의고사 자연계열Ⅱ 수학 1·2) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KNU-M25B"
WIDTH = 2
RESULTS = []
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return [sp.sympify(s.strip()) for s in str(v).split(";")]


def p01():
    """(Ⅰ)에서 b=0 → f(2a)=f(a)√(4+f(a)^2), a=h → f'(b)=√(4+f(b)^2), f(0)=0 : 미분방정식을 수치로 풀어 독립 재계산."""
    oks, msgs = [], []
    # f(0): a=b=0 대입식 2f(0)(2+f(0)^2)=0 의 실근
    y = sp.symbols("y", real=True)
    f0 = sp.solve(sp.Eq(2 * y * (2 + y ** 2), 0), y)
    # 초기값 문제 f'=√(4+f^2), f(0)=0 을 수치 적분(odefun) — 닫힌 해를 쓰지 않음
    F = mp.odefun(lambda x, u: mp.sqrt(4 + u ** 2), 0, 0)
    a0, a1 = load_answer(1, 0)
    ok1 = f0 == [a0] and abs(F(mp.log(4)) - sp.N(a1, 30)) < 1e-15
    # 그 해가 (Ⅱ) f(ln2)=3/2, (Ⅲ) f'(0)=2 와 조건 (Ⅰ)을 만족하는지 무작위 점에서 확인
    ok1 &= abs(F(mp.log(2)) - mp.mpf(3) / 2) < 1e-15 and abs(mp.sqrt(4 + F(0) ** 2) - 2) < 1e-20
    import random
    random.seed(3)
    for _ in range(8):
        a = mp.mpf(random.uniform(0, 0.6)); b = mp.mpf(random.uniform(0, 0.6))
        lhs = 2 * F(2 * a + b) - 2 * F(b)
        rhs = F(a) * mp.sqrt(4 + F(a) ** 2) * mp.sqrt(4 + F(b) ** 2) + F(a) ** 2 * F(b)
        ok1 &= abs(lhs - rhs) < 1e-12
    oks.append(bool(ok1)); msgs.append(f"f(0) 해={f0}, f(ln4)={mp.nstr(F(mp.log(4)),15)}, (Ⅰ) 무작위 8점 성립")
    # 1-2: f'(ln2) 를 수치 미분
    d = mp.diff(F, mp.log(2))
    (a2,) = load_answer(1, 1)
    oks.append(abs(d - sp.N(a2, 30)) < 1e-10); msgs.append(f"f'(ln2)={mp.nstr(d,15)}")
    # 1-3: ∫{f'}^2 수치 적분
    I = mp.quad(lambda x: 4 + F(x) ** 2, [0, mp.log(2)])   # {f'}^2 = 4+f^2 (1-2 의 결과)
    (a3,) = load_answer(1, 2)
    oks.append(abs(I - sp.N(a3, 30)) < 1e-10); msgs.append(f"∫(f')^2={mp.nstr(I,15)} md={sp.N(a3,15)}")
    report(1, all(oks), "; ".join(msgs))


def _geom(p):
    """좌표로 직접 작도: D 원점, A(√3,0), B(√3,3), C(0,3), F(0,2), E(√3,2), P(-PD,0)."""
    p = mp.mpf(p)
    PD = 2 / mp.tan(p)                       # ∠FPD=p 를 만족하는 P (DF=2)
    r = 2 / PD                               # 호 DD' 의 길이 2
    P = mp.matrix([-PD, 0]); A = mp.matrix([mp.sqrt(3), 0]); B = mp.matrix([mp.sqrt(3), 3]); D = mp.matrix([0, 0])
    rot = lambda X: P + mp.matrix([[mp.cos(r), -mp.sin(r)], [mp.sin(r), mp.cos(r)]]) * (X - P)
    return P, A, B, D, rot, r


def S_geom(p):
    """경계 A→B→(호)B'→A'→D'→(호)D→A 를 따라 그린 정리 ½∮(x dy - y dx)."""
    P, A, B, D, rot, r = _geom(p)
    Bp, Ap, Dp = rot(B), rot(A), rot(D)
    seg = lambda U, V: (U[0] * V[1] - U[1] * V[0]) / 2
    def arc(R0, th0, th1):                   # P 중심 반지름 R0, 각 th0→th1
        x = lambda t: P[0] + R0 * mp.cos(t); yy = lambda t: P[1] + R0 * mp.sin(t)
        return mp.quad(lambda t: (x(t) * R0 * mp.cos(t) + yy(t) * R0 * mp.sin(t)) / 2, [th0, th1])
    thB = mp.atan2(B[1] - P[1], B[0] - P[0]); RB = mp.norm(B - P); RD = mp.norm(D - P)
    return seg(A, B) + arc(RB, thB, thB + r) + seg(Bp, Ap) + seg(Ap, Dp) + arc(RD, r, 0) + seg(D, A)


def p02():
    oks, msgs = [], []
    pv = [mp.pi / 8, mp.pi / 6, mp.mpf("0.7"), mp.pi / 3]
    (r_ans,) = load_answer(2, 0)
    rs = sp.lambdify(sp.Symbol("p"), r_ans, "mpmath")
    ok1 = all(abs(_geom(p)[5] - rs(p)) < 1e-25 for p in pv)
    oks.append(ok1); msgs.append(f"r(p) 4점={ok1}")
    (S_ans,) = load_answer(2, 1)
    Ss = sp.lambdify(sp.Symbol("p"), S_ans, "mpmath")
    errs = [abs(S_geom(p) - Ss(p)) for p in pv]
    ok2 = max(errs) < 1e-20
    oks.append(ok2); msgs.append(f"S(p) 그린정리 4점 최대오차={mp.nstr(max(errs),3)}")
    # 2-3: q=∠EPD=atan(2/PA), PA=PD+√3 → 수치 미분
    q = lambda p: mp.atan(2 / (_geom(p)[1][0] - _geom(p)[0][0]))
    dq = mp.diff(q, mp.pi / 6)
    (a3,) = load_answer(2, 2)
    oks.append(abs(dq - sp.N(a3, 30)) < 1e-20); msgs.append(f"dq/dp={mp.nstr(dq,15)} md={a3}")
    # 2-4: S(p(t)) 를 기하 넓이로 수치 적분
    pt = lambda t: mp.pi / 3 * t if t < 1 else (mp.pi / 3 if t < 2 else mp.pi - mp.pi / 3 * t)
    mp.mp.dps = 20
    I = mp.quad(lambda t: S_geom(pt(t)), [0.5, 1, 2, 2.5])
    m_, n_ = load_answer(2, 3)
    ok4 = abs(I - (int(m_) * mp.sqrt(3) + int(n_) * mp.log(3) / mp.pi)) < 1e-12
    mp.mp.dps = 30
    oks.append(ok4); msgs.append(f"∫S(p(t))dt={mp.nstr(I,15)} (m,n)=({m_},{n_})")
    report(2, all(oks), "; ".join(msgs))


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
