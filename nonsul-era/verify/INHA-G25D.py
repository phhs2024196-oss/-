#!/usr/bin/env python3
"""INHA-G25D (2025학년도 인하대 자연계열 논술 오전·의예과; [문제 1]은 INHA-G25A-03 과 동일해 생략) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp
from fractions import Fraction as Fr

PREFIX = "INHA-G25D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")




def _next_term(a, n):
    """a_n=a·π(분수 계수) → a_{n+1}: sin(2x-3a/2)=cos(x-3a/2) 의 양의 해를 정의에서 직접 수치 판정 없이
    sin A = sin B 의 두 가족으로 열거(B=π/2-(x-3a/2)), 중복 제거 후 (n+1)번째."""
    sols = set()
    # 2x-3a/2 = π/2-x+3a/2 + 2kπ  →  x = a + 1/6 + 2k/3 ;  2x-3a/2 = π-(π/2-x+3a/2)+2kπ → x = 1/2 + 2k
    for k in range(-3 * (n + 6), 3 * (n + 6)):
        for xv in (a + Fr(1, 6) + Fr(2 * k, 3), Fr(1, 2) + 2 * k):
            if xv > 0:
                sols.add(xv)
    return sorted(sols)[n]


def _seq(a1, N):
    a = [Fr(a1)]
    for n in range(1, N):
        a.append(_next_term(a[-1], n))
    return a


def _sin_classes(vals):
    """sin(uπ) 값의 서로 다른 개수: sin uπ = sin vπ ⇔ u≡v (mod 2) 또는 u+v≡1 (mod 2)."""
    reps = []
    for u in vals:
        if not any(((u - v) % 2 == 0) or ((u + v - 1) % 2 == 0) for v in reps):
            reps.append(u)
    return len(reps)


def p02():
    """(2-1) a1=0 → a2,a3 (2-2) a2=a1+π/6 인 a1 전수(분모 720 격자) (2-3) sin 집합 원소 개수 전수."""
    import sympy as sp
    s = _seq(0, 3)
    ok1 = (s[1], s[2]) == (Fr(1, 2), Fr(4, 3))
    D = 720
    good = [Fr(k, D) for k in range(0, 2 * D) if _seq(Fr(k, D), 2)[1] == Fr(k, D) + Fr(1, 6)]
    exp = [Fr(k, D) for k in range(0, 2 * D) if (Fr(1, 3) < Fr(k, D) <= Fr(1, 2)) or Fr(k, D) == 1]
    ok2 = good == exp
    cnts = {}
    for a1 in good:
        if a1 + Fr(1, 6) <= Fr(2, 3):
            c = _sin_classes(_seq(a1, 60))
            cnts.setdefault(c, []).append(a1)
    ok3 = sorted(cnts) == [3, 4, 5] and cnts[3] == [Fr(5, 12)] and cnts[5] == [Fr(1, 2)]
    report(2, ok1 and ok2 and ok3,
           f"(2-1) a2={s[1]}π a3={s[2]}π (2-2) 조건 만족 a1/π 격자 {len(good)}개 = (1/3,1/2]∪{{1}} (2-3) 개수 {sorted(cnts)} (3: a1={cnts.get(3)}π, 5: a1={cnts.get(5)}π)")


def _sigma(S):
    S = sorted(S)
    st = set(S)
    return sum(1 for i in range(len(S)) for j in range(i + 1, len(S)) if S[i] + S[j] in st)


def p03():
    """(3-1) σ 직접 계산 (3-2) n=2,3 전수 [증명-수치확인] (3-3) 축소판 전수로 '두 형태' 구조 확인 + 원래 개수 공식."""
    import itertools
    s1 = _sigma(range(2, 21, 2))
    ok1 = s1 == 20
    ok2 = True
    msg2 = []
    for n_, N in ((2, 16), (3, 15)):
        best_pairs = best_sig = 0
        argmax1 = []
        for S in itertools.combinations(range(1, N + 1), 2 * n_ + 1):
            x_, y_ = S[-1], S[-2]
            pr = sum(1 for i in range(len(S)) for j in range(i + 1, len(S)) if S[i] + S[j] in (x_, y_))
            best_pairs = max(best_pairs, pr)
            sg = _sigma(S)
            if sg > best_sig:
                best_sig, argmax1 = sg, []
            if sg == best_sig and S[0] == 1:
                argmax1.append(S)
        ok2 &= best_pairs == 2 * n_ - 1 and best_sig == n_**2 and argmax1 == [tuple(range(1, 2 * n_ + 2))]
        msg2.append(f"n={n_}: 쌍최대 {best_pairs}, M={best_sig}, 1∈S 극값집합 {argmax1}")
    # (3-3) 축소판: n(S)=2m+2, σ=m^2+m, S⊂{1..N} → 두 형태 {d..(2m+1)d,(2m+2)d}, {d..(2m+1)d,(2m+3)d} 뿐인지 전수
    ok3 = True
    for m_, N in ((2, 26), (3, 20)):
        found = {S for S in itertools.combinations(range(1, N + 1), 2 * m_ + 2) if _sigma(S) == m_ * m_ + m_}
        exp = set()
        for d in range(1, N + 1):
            base = tuple(d * k for k in range(1, 2 * m_ + 2))
            for top in ((2 * m_ + 2) * d, (2 * m_ + 3) * d):
                if top <= N:
                    exp.add(base + (top,))
        ok3 &= found == exp
    full = _sigma(range(1, 203)) == 10100 and _sigma(list(range(1, 202)) + [203]) == 10100
    cnt = sum(1 for d in range(1, 2025) if 202 * d <= 2024) + sum(1 for d in range(1, 2025) if 203 * d <= 2024)
    ok3 &= full and cnt == 19
    report(3, ok1 and ok2 and ok3,
           f"(3-1) σ={s1} (3-2) {'; '.join(msg2)} [증명-수치확인] (3-3) 축소판 m=2,3 두 형태 일치, 본 문항 개수 {cnt}")


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
