#!/usr/bin/env python3
"""KNU-G25A (경북대 2025 수시 논술 자연계열Ⅰ 수학 1·2) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KNU-G25A"
WIDTH = 2
RESULTS = []
mp.mp.dps = 40


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    """소문항 sub(0부터)의 answer.value 를 ';' 로 나눠 sympy 식 목록으로."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return [sp.sympify(s.strip()) for s in str(v).split(";")]


def p01():
    k, i = sp.symbols("k i")
    oks, msgs = [], []
    # (1-1) θ: cos(θ+π/6)=(3√5-1)/8, 0<θ<π/2 → 수선의 발을 실제 좌표로 반복해 h_k 를 구함
    th = mp.acos((3 * mp.sqrt(5) - 1) / 8) - mp.pi / 6
    assert 0 < th < mp.pi / 2
    def hs(L, K=60):
        O = mp.matrix([0, 0]); u = mp.matrix([1, 0]); w = mp.matrix([mp.cos(th), mp.sin(th)])
        B = L * w; out = []
        for _ in range(K):
            A = (B.T * u)[0] * u            # B 에서 OA 에 내린 수선의 발
            out.append(mp.norm(B - A))
            B = (A.T * w)[0] * w            # A 에서 OB 에 내린 수선의 발
        return out
    S1 = mp.fsum(mp.sqrt(h) for h in hs(1, 4000))
    a, b = load_answer(1, 0)
    oks.append(abs(S1 - (float(a) + float(b) * mp.sqrt(15))) < 1e-25)
    # 자연수 p,q 로 S1=p+q√15 인 것이 (ㄱ),(ㄴ) 하나뿐인지: q 는 (S1-p)/√15 정수
    cand = [(p, int(mp.nint((S1 - p) / mp.sqrt(15)))) for p in range(1, 40)
            if abs((S1 - p) / mp.sqrt(15) - mp.nint((S1 - p) / mp.sqrt(15))) < 1e-20 and mp.nint((S1 - p) / mp.sqrt(15)) > 0]
    oks.append(cand == [(int(a), int(b))])
    msgs.append(f"S1={mp.nstr(S1,15)} 후보={cand}")
    # S_2 = 2 S_1 확인(OB1=j^2)
    S2 = mp.fsum(mp.sqrt(h) for h in hs(4, 4000))
    oks.append(abs(S2 - 2 * S1) < 1e-25)
    # (1-2) 등식 n=1..40, (ㄷ)(ㄹ)(ㅁ) 를 증명 틀에 대입해 k=1..40 확인
    c, r, m = load_answer(1, 1)
    ok2 = sp.Rational(1) - sp.Rational(1, 2) == c
    for n in range(1, 41):
        lhs = sum(sp.Rational((-1) ** (t + 1), t) for t in range(1, 2 * n + 1))
        ok2 &= lhs == sum(sp.Rational(1, n + t) for t in range(1, n + 1))
    for kv in range(1, 41):
        mid = sum(sp.Rational(1, kv + t) for t in range(1, kv + 1)) + sp.Rational(1, 2 * kv + 1) - sp.Rational(1, 2 * kv + 2)
        step3 = sum(sp.Rational(1, kv + t) for t in range(2, kv + 2)) + 1 / r.subs(k, kv)
        step4 = sum(1 / m.subs({k: kv, i: t}) for t in range(1, kv + 2))
        target = sum(sp.Rational(1, kv + 1 + t) for t in range(1, kv + 2))
        ok2 &= (mid == step3 == step4 == target)
    oks.append(bool(ok2)); msgs.append(f"(ㄷ,ㄹ,ㅁ)=({c},{r},{m}) k=1..40 성립={bool(ok2)}")
    # (1-3) 교대합 극한을 직접 수치 계산: S_j = j*S1
    L = 2 / S1 * mp.nsum(lambda j: (-1) ** (int(j) + 1) / j, [1, mp.inf])   # S_j=jS_1 (위에서 S_2=2S_1 확인)
    val = mp.e ** (L + mp.sqrt(15) * mp.log(2))
    (ans3,) = load_answer(1, 2)
    oks.append(abs(val - int(ans3)) < 1e-20)
    msgs.append(f"exp(극한)={mp.nstr(val,15)} md={ans3}")
    report(1, all(oks), "; ".join(msgs))


def _inradius(P, Q, R):
    a = mp.norm(Q - R); b = mp.norm(P - R); c = mp.norm(P - Q)
    s = (a + b + c) / 2
    area = abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2
    return area / s


def _angle(V, P, Q):   # 꼭짓점 V 에서 VP, VQ 사이 각
    u = P - V; w = Q - V
    return mp.acos((u.T * w)[0] / (mp.norm(u) * mp.norm(w)))


def _fgh(t):
    """조건 (Ⅰ)(Ⅱ)로 a1<a2<a3 를 이분법으로 직접 구해 f,g,h 를 계산."""
    B = mp.matrix([t, mp.sqrt(8 * t + 16)]); prev = mp.matrix([0, 0]); vals = []
    for _ in range(3):
        F = lambda a: _inradius(prev, mp.matrix([a, 0]), B) - 1
        lo, hi = prev[0] + mp.mpf("1e-30"), prev[0] + 1
        while F(hi) < 0:
            hi = prev[0] + 2 * (hi - prev[0])
        A = mp.matrix([mp.findroot(F, (lo, hi), solver="bisect" if False else "anderson"), 0])
        vals.append(mp.tan(_angle(A, mp.matrix([0, 0]), B) / 2))
        prev = A
    return vals


def p02():
    oks, msgs = [], []
    # (2-1) 무작위 삼각형 20개로 (1)(2) 수치 확인 [증명-수치확인]
    import random
    random.seed(1)
    ok1 = True
    for _ in range(20):
        al = mp.mpf(random.uniform(0.2, 2.0)); be = mp.mpf(random.uniform(0.2, 3.0 - float(al)))
        A = mp.matrix([0, 0]); Bp = mp.matrix([random.uniform(1, 5), 0])
        # C: 두 밑각 al, be 로 결정
        L = Bp[0]; x = L * mp.tan(be) / (mp.tan(al) + mp.tan(be)); C = mp.matrix([x, x * mp.tan(al)])
        q = C[1]; r = _inradius(A, Bp, C); S = L * q / 2
        ca, cb = mp.cot(al / 2), mp.cot(be / 2)
        ok1 &= abs(S - r * q / 2 * (ca + cb)) < 1e-25 and abs(S - r ** 2 * (ca + cb + (ca + cb) / (ca * cb - 1))) < 1e-25
    oks.append(ok1); msgs.append(f"2-1 무작위 20개 [증명-수치확인]={ok1}")
    # (2-2) t 여러 값에서 g/f = h/g [증명-수치확인]
    ok2 = True
    for t in [-1, -0.5, 0, 0.7, 2, 5, 13]:
        f, g, h = _fgh(mp.mpf(t))
        ok2 &= abs(g / f - h / g) < 1e-20
    oks.append(ok2); msgs.append(f"2-2 t 7점 g/f=h/g [증명-수치확인]={ok2}")
    # (2-3) 기하로부터 직접 구한 h(t) 를 수치 미분
    hd = mp.diff(lambda t: _fgh(t)[2], 0)
    (ans,) = load_answer(2, 2)
    oks.append(abs(hd - sp.N(ans, 40)) < 1e-15); msgs.append(f"h'(0)={mp.nstr(hd,15)} md={ans}")
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
