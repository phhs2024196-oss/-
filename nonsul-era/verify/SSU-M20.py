#!/usr/bin/env python3
"""숭실대 2020학년도 모의논술 (SSU-M20-01) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수(번호 = 원문 문제 번호)를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M20"
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
    """(1) 교점·기울기 일치 연립방정식을 여러 시작점에서 nsolve, (2) 원의 아래 호와 max(0, ln|x|+1) 사이를 수치적분."""
    import mpmath as mp
    mp.mp.dps = 30
    a, x1, y1 = sp.symbols("a x1 y1", real=True)
    eqs = [x1 ** 2 + (y1 - a) ** 2 - 2, y1 - sp.log(x1) - 1, x1 ** 2 - (a - y1)]   # 기울기 x1/(a-y1)=1/x1
    sols = set()
    for st in [(0.5, 0.5, 1), (1, 1, 2), (2, 1.5, 3), (0.8, 0.7, 1.5), (1.3, 1.2, 2.5), (3, 2, 5)]:
        try:
            s_ = sp.nsolve(eqs, (x1, y1, a), st, prec=30)
            if s_[0] > 0:
                sols.add(round(float(s_[2]), 12))
        except Exception:  # noqa: BLE001
            pass
    ans1 = load_answer(1, 0)
    ok1 = sols == {float(ans1)}
    A = float(ans1)
    top = lambda X: A - mp.sqrt(2 - X * X)                            # 원의 아래쪽 호
    bot = lambda X: max(mp.mpf(0), mp.log(abs(X)) + 1) if X != 0 else mp.mpf(0)
    S_num = mp.quad(lambda X: top(X) - bot(X), [-1, -1 / mp.e, 0, 1 / mp.e, 1])
    nonneg = all(top(X) - bot(X) >= -1e-20 for X in mp.linspace(-1, 1, 201))
    ans2 = load_answer(1, 1)
    ok2 = nonneg and abs(S_num - mp.mpf(str(sp.N(ans2, 40)))) < 1e-20
    report(1, ok1 and ok2, f"(1) a 후보={sols} (2) S={mp.nstr(S_num, 15)}, md={ans1}, {ans2}")


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
