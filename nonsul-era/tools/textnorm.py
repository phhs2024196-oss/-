"""원본 대조·중복 대조 공용 정규화.

원칙(신뢰성): 정규화는 '표기만 다르고 내용이 같은 것'만 같게 만든다.
글자·숫자·변수·부호처럼 내용이 바뀌는 차이는 절대 지우지 않는다.
"""
from __future__ import annotations

import collections
import pathlib
import re
import unicodedata

import yaml

# DB 에만 있는 주석(원본에 없는 글): [판독확인: …] [그림 n: …] 【…】 출처 표기 등
ANNOT = re.compile(r"\[(?:판독확인|편집)[^\]]*\]")
FIG = re.compile(r"\[(그림|표)\s*(\d*)\s*:[^\]]*\]")      # [그림 1: 설명] → [그림 1] (원본엔 번호만 있음)
LABELS = re.compile(r"\[\s*(?:문항|문제)\s*[\d\-–]+\s*\]|\(\s*[가-하]\s*\)|<\s*[가-하]\s*>|【\s*[가-하]\s*】")  # 번호·제시문 표지
POINTS = re.compile(r"[(\[]\s*\d+\s*점\s*[)\]]")   # 배점 표기는 따로 대조하므로 본문 비교에서 뺀다
LATEX_WORD = {
    r"\le": "≤", r"\leq": "≤", r"\leqq": "≤", r"\ge": "≥", r"\geq": "≥", r"\geqq": "≥", r"\ne": "≠",
    r"\neq": "≠", r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\mp": "∓", r"\infty": "∞",
    r"\int": "∫", r"\sum": "Σ", r"\lim": "lim", r"\sqrt": "√", r"\to": "→", r"\rightarrow": "→",
    r"\Rightarrow": "⇒", r"\iff": "⇔", r"\Leftrightarrow": "⇔", r"\in": "∈", r"\cup": "∪",
    r"\cap": "∩", r"\subset": "⊂", r"\angle": "∠", r"\triangle": "△", r"\circ": "∘", r"\prime": "′",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\theta": "θ", r"\pi": "π",
    r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\omega": "ω", r"\phi": "φ", r"\varphi": "φ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\Delta": "Δ", r"\cdots": " ", r"\ldots": " ", r"\dots": " ", r"\vdots": " ",
    r"\ell": "l", r"\mid": "|", r"\vert": "|", r"\lvert": "|", r"\rvert": "|", r"\Vert": "‖",
    r"\perp": "⊥", r"\parallel": "∥", r"\equiv": "≡", r"\approx": "≈", r"\sim": "∼", r"\propto": "∝",
    r"\varnothing": "∅", r"\emptyset": "∅", r"\notin": "∉", r"\supset": "⊃", r"\subseteq": "⊆",
    r"\leftarrow": "←", r"\longrightarrow": "→", r"\mapsto": "→", r"\partial": "∂", r"\nabla": "∇",
    r"\sec": "sec", r"\csc": "csc", r"\cot": "cot", r"\min": "min", r"\max": "max", r"\gcd": "gcd",
    r"\arcsin": "arcsin", r"\cosh": "cosh", r"\sinh": "sinh", r"\bmod": "mod", r"\pmod": "mod",
    r"\ll": "≪", r"\gg": "≫", r"\div": "÷", r"\eta": "η", r"\Sigma": "Σ", r"\prod": "Π",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉", r"\bullet": "•",
    r"\therefore": "∴", r"\because": "∵", r"\square": "□", r"\supseteq": "⊇", r"\nearrow": "↗",
    r"\searrow": "↘", r"\lessgtr": "≶", r"\gtrless": "≷",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\ln": "ln", r"\log": "log", r"\exp": "exp",
}
# 표기만 바꾸는 명령: 지운다(내용 인자는 남김)
LATEX_DROP = re.compile(r"\\(?:d|t)?frac|\\boxed|\\dot|\\widehat|\\mathbb|\\middle|\\overset|\\underset|\\not|\\cfrac|\\binom|\\left|\\right|\\displaystyle|\\mathrm|\\mathbf|\\boldsymbol|"
                        r"\\overline|\\overrightarrow|\\vec|\\bar|\\hat|\\text|\\textrm|\\operatorname|"
                        r"\\quad|\\qquad|\\[,;:! ]|\\limits|\\big|\\Big|\\bigg|\\Bigg|~")
# 한글 수식편집기(HWP) 스크립트
HWP_WORD = {"over": " ", "LEQ": "≤", "leq": "≤", "le": "≤", "GEQ": "≥", "geq": "≥", "ge": "≥",
            "times": "×", "TIMES": "×", "cdot": "·", "CDOTS": "…", "cdots": "…", "LDOTS": "…",
            "sqrt": "√", "SQRT": "√", "int": "∫", "INT": "∫", "sum": "Σ", "SUM": "Σ", "pi": "π", "PI": "π",
            "theta": "θ", "alpha": "α", "beta": "β", "inf": "∞", "INF": "∞", "lim": "lim", "LIM": "lim",
            "rm": " ", "it": " ", "bar": " ", "vec": " ", "left": " ", "right": " ", "rarrow": "→",
            "->": "→", "ne": "≠", "NEQ": "≠", "angle": "∠", "triangle": "△", "pm": "±", "PLUSMINUS": "±",
            "lambda": "λ", "gamma": "γ", "delta": "δ", "sigma": "σ", "omega": "ω", "prime": "′"}
OPS = set("÷≪≫Π⌊⌋⌈⌉•∴∵□⊇↗↘≶≷=+-<>≤≥≠×·±∓√∫Σ→←⇒⇔∈∉∪∩⊂⊃⊆∠△°∘′!|‖()[],:⦃⦄ℓ⊥∥≡≈∼∝∅∂∇¿")   # ^ / 는 조판 차이라 제외
GREEK = set("αβγδθπλμσωφεΔ∞…")
BRACKETS = str.maketrans({"【": "[", "】": "]", "〔": "[", "〕": "]", "［": "[", "］": "]", "⋯": " ", "…": " "})
MINUS = dict.fromkeys(map(ord, "−–—‐‑﹣－"), "-")


ANNOT_HEAD = re.compile(r"\[\s*(?:(판독확인|편집)|(그림|표)\s*([\d\-–]*)\s*:)")


def strip_annot(s: str) -> str:
    """DB 에만 있는 주석 제거: [판독확인: …] → 삭제, [그림 1-2: 설명] → [그림 1-2] (원본엔 번호만).
    설명 안에 [0,1] 같은 대괄호가 있어도 짝을 세어 끝을 찾는다."""
    out, i = [], 0
    while True:
        m = ANNOT_HEAD.search(s, i)
        if not m:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:m.start()])
        depth, j = 0, m.start()
        while j < len(s):
            depth += {"[": 1, "]": -1}.get(s[j], 0)
            j += 1
            if depth == 0:
                break
        if m.group(2) and m.group(3):
            out.append(f" [{m.group(2)} {m.group(3)}] ")
        else:
            out.append(" ")
        i = j


def latex_plain(s: str) -> str:
    s = strip_annot(s)
    s = s.replace("\\{", " ⦃ ").replace("\\}", " ⦄ ").replace("\\lbrace", " ⦃ ").replace("\\rbrace", " ⦄ ")
    s = re.sub(r"\\(?:begin|end)\{[a-z*]+\}", " ", s).replace("&", " ").replace("\\\\", " ")
    s = re.sub(r"\\text\{([^}]*)\}", r" \1 ", s)
    s = re.sub(r"\\(?:mathrm|operatorname)\{([^}]*)\}", r" \1 ", s)
    for w in sorted(LATEX_WORD, key=len, reverse=True):
        s = re.sub(re.escape(w) + r"(?![A-Za-z])", " " + LATEX_WORD[w] + " ", s)
    s = LATEX_DROP.sub(" ", s)
    s = re.sub(r"\\[A-Za-z]+", " ¿ ", s)    # 모르는 명령은 지우지 않고 ¿ 로 남겨 차이로 드러나게(신뢰성)
    return s.replace("$", " ").replace("{", " ").replace("}", " ").replace("_", " ").replace("^", " ^ ")


def hwp_plain(s: str) -> str:
    s = s.replace("`", " ").replace("~", " ")
    s = re.sub(r"[A-Za-z]+|->", lambda m: " " + HWP_WORD.get(m.group(0), m.group(0)) + " ", s)
    return s.replace("{", " ").replace("}", " ").replace("$", " ").replace("_", " ").replace("^", " ^ ")


PUA_PATH = pathlib.Path(__file__).resolve().parent.parent / "db" / "pua_map.yaml"
_PUA: dict | None = None


def pua_map() -> dict:
    """PDF 수식 글꼴(HyhwpEQ 등)의 사용자 정의 영역 글자 → 실제 글자. tools/pua_learn.py 로 학습·누적."""
    global _PUA
    if _PUA is None:
        d = yaml.safe_load(PUA_PATH.read_text(encoding="utf-8")) if PUA_PATH.exists() else {}
        _PUA = {chr(int(k, 16)): v["char"] for k, v in (d or {}).get("map", {}).items()}
    return _PUA


def is_pua(ch: str) -> bool:
    return 0xE000 <= ord(ch) <= 0xF8FF


def plain(s: str, kind: str = "latex") -> str:
    s = s or ""
    if kind == "pdf":
        m = pua_map()
        s = "".join(m.get(ch, ch) for ch in s)          # 모르는 PUA 글자는 그대로 남겨 FLAG 되게 한다
    s = s.translate(BRACKETS)
    s = unicodedata.normalize("NFKC", s).translate(MINUS).replace("'", "′").replace("’", "′")   # 𝑥→x −→- ℓ→l f'→f′
    s = LABELS.sub(" ", s)          # [문항 1] [문제 1-2] (가) 같은 표지는 양쪽에서 뺀다(번호는 ID·배점으로 따로 확인)
    if kind == "latex":
        s = latex_plain(s)
    elif kind == "hwp":
        s = hwp_plain(s)
    else:
        s = s.replace("{", "⦃").replace("}", "⦄")
    return s


def hangul(s: str) -> str:
    return "".join(ch for ch in s if "가" <= ch <= "힣")


def math_seq(s: str) -> list[str]:
    """내용 기호열(순서 유지): 숫자 한 자리씩, 라틴 문자 한 글자씩, 그리스·연산 기호.
    숫자를 자리 단위로 쪼개는 이유: 원본 PDF 에서 3^2 가 '32' 로 붙어 나와도 같게 보기 위해."""
    return [ch for ch in s if ch.isdigit() and ch.isascii() or ch.isascii() and ch.isalpha()
            or ch in OPS or ch in GREEK or is_pua(ch)]


def math_tokens(s: str) -> collections.Counter:
    return collections.Counter(t for t in math_seq(s) if t != ".")
