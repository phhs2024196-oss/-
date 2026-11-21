#!/usr/bin/env python3
"""숭실대 SSU-G18-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G18"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answers(seq):
    """소문항 id → answer.value(문자열). 증명 소문항은 None."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return {str(s["id"]): s["answer"]["value"] for s in d["subparts"]}


def p01():
    """원 중심 (0,2) 반지름 2, 부채꼴 반지름 2√3 — 겹치는 넓이를 극좌표 적분으로 직접 계산해 S1-S2 를 수치 최대화."""
    import mpmath as mp
    mp.mp.dps = 30
    ans = sp.sympify(load_answers(1)["1"])
    # 원 r=4 sinφ (극좌표). 부채꼴 0≤φ≤θ, r≤2√3. 겹침 = ∫_0^θ ½ min(4 sinφ, 2√3)^2 dφ
    R = 2 * mp.sqrt(3)
    S2 = lambda th: mp.quad(lambda p: 0.5 * min(4 * mp.sin(p), R) ** 2, [0, th])
    T = lambda th: (0.5 * R**2 * th - S2(th)) - S2(th)
    th0 = mp.findroot(lambda s: mp.diff(T, s), 0.6)
    grid = max((T(mp.pi/3*k/600), k) for k in range(601))
    ok = abs(mp.cos(2*th0) - float(ans)) < 1e-12 and abs(mp.pi/3*grid[1]/600 - th0) < 0.01
    # 잘못 읽은 중심 (2,0) 이면 θ=0 에서 현 길이 4 > 2√3 → 그림과 모순 확인
    report(1, ok, f"θ*={mp.nstr(th0,12)}, cos2θ*={mp.nstr(mp.cos(2*th0),15)}, md={ans}")


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
