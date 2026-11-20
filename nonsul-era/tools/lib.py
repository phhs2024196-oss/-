"""논술의 시대 문항 DB 공용 함수 (양식 v2: 문항 1개 = YAML 파일 1개).

양식 정의: db/item_template.yaml, 02_문항DB_저장계획서.md §5
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import yaml

KIT = pathlib.Path(__file__).resolve().parent.parent          # nonsul-era/
TAXONOMY_PATH = KIT / "db" / "taxonomy.yaml"
SCHEMA_VERSION = "2.0"
STATUS_ORDER = ["draft", "r1_pass", "r2_pass", "verified", "sampled", "published"]

ID_ORIGINAL = re.compile(r"^(?P<school>[A-Z]{2,4})-(?P<type>T\d{2})-(?P<seq>\d{3})$")
ID_PAST = re.compile(r"^(?P<school>[A-Z]{2,4})-(?P<src>[GM])(?P<yy>\d{2})(?P<session>[A-Z]?)-(?P<no>\d{2})$")


# ── YAML 입출력: 여러 줄 문자열은 | 블록으로 써서 LaTeX 역슬래시를 그대로 둔다 ──
class _Dumper(yaml.SafeDumper):
    pass


def _str_rep(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_rep)


def dump_yaml(obj) -> str:
    return yaml.dump(obj, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=1000)


def load_taxonomy() -> dict:
    return yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8"))


def load_catalog(name: str) -> dict:
    """db/misconceptions.yaml, db/concepts.yaml → {id: entry}"""
    p = KIT / "db" / f"{name}.yaml"
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {e["id"]: e for e in data.get(name, []) if isinstance(e, dict) and e.get("id")}


def load_item(path: pathlib.Path) -> dict:
    item = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise ValueError("YAML 최상위가 매핑이 아님")
    return item


def item_files() -> list[pathlib.Path]:
    return sorted((KIT / "items").rglob("*.yaml"))


def all_item_ids() -> set[str]:
    return {p.stem for p in item_files()}


def item_seq(item_id: str) -> int | None:
    m = ID_ORIGINAL.match(item_id or "")
    return int(m["seq"]) if m else None


def load_books() -> list[tuple[pathlib.Path, dict]]:
    out = []
    for bf in sorted((KIT / "books").glob("*.yaml")):
        if bf.name.endswith(".lock.yaml"):
            continue
        out.append((bf, yaml.safe_load(bf.read_text(encoding="utf-8")) or {}))
    return out


# ── 본문 텍스트 모으기(수식 검사·검색·중복 탐지용) ───────────────────────────
def item_texts(item: dict) -> dict[str, str]:
    """{'prompt': 학생에게 보이는 문제, 'solution': 해설 전체, 'rubric': 채점 근거 전체}"""
    prompt = [item.get("stem") or ""]
    prompt += [s.get("body") or "" for s in item.get("stimuli") or []]
    sol, rub = [], []
    for sp in item.get("subparts") or []:
        prompt.append(sp.get("stem") or "")
        for so in sp.get("solutions") or []:
            sol.append(so.get("key_idea") or "")
            sol += [st.get("content") or "" for st in so.get("steps") or []]
        for c in (sp.get("rubric") or {}).get("criteria") or []:
            rub.append(c.get("evidence") or "")
    return {"prompt": "\n".join(prompt), "solution": "\n".join(sol), "rubric": "\n".join(rub)}


CONTENT_KEYS = ("stem", "stimuli", "subparts", "points", "level")


def item_content_hash(item: dict) -> str:
    """교재에 보이는 내용(문제·소문항·해설·채점표·정답·배점·난이도)만 해시. status·reviews·updated 는 제외."""
    visible = {k: item.get(k) for k in CONTENT_KEYS}
    return hashlib.sha256(json.dumps(visible, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ── 2028 출제범위 추정 (키워드 휴리스틱: 확정이 아니라 태깅 우선순위용) ──────────
GEOM_PAT = re.compile(r"\\overrightarrow|벡터|내적|타원|쌍곡선|준선|정사영|이면각|삼수선|좌표공간|공간좌표")
TRANS_PAT = re.compile(r"\\ln|\\log|e\^|\\sin|\\cos|\\tan|\\sec|\\csc|\\cot")
CALC_PAT = re.compile(r"f'|\\int|미분|적분|접선|극값|극대|극소|도함수|넓이|부피")
CALC2_KW = re.compile(
    r"치환적분|부분적분|역함수의 미분|음함수|매개변수|덧셈정리|이계도함수|변곡점|곡선의 길이|"
    r"급수|\\lim_\{n\s*\\to\s*\\infty\}|\\sum_\{[a-z]=1\}\^\{?\\infty|속도|가속도|구분구적"
)
STAT_TOOL_PAT = re.compile(r"확률|기댓값|분산|표준편차|정규분포|이항분포|표본평균|확률변수")
STAT_PAT = re.compile(r"확률|경우의 수|조합|순열|기댓값|분산|표본|정규분포|이항분포|\\binom|{}_\{?\d*\}?C")


def guess_scope_2028(text: str) -> tuple[str, list[str]]:
    """반환: (scope 태그, 2015 개정 과목 추정 목록)."""
    subjects = set()
    if GEOM_PAT.search(text):
        subjects.add("기하")
    if CALC2_KW.search(text) or (TRANS_PAT.search(text) and CALC_PAT.search(text)):
        subjects.add("미적분")
    if STAT_PAT.search(text):
        subjects.add("확률과통계")
    if CALC_PAT.search(text) and "미적분" not in subjects:
        subjects.add("수학Ⅱ")
    if TRANS_PAT.search(text) or re.search(r"\\sum|수열|a_\{?n", text):
        subjects.add("수학Ⅰ")
    scope = "geom" if "기하" in subjects else "calc2" if "미적분" in subjects else "common"
    order = ["수학", "수학Ⅰ", "수학Ⅱ", "미적분", "확률과통계", "기하"]
    return scope, [s for s in order if s in subjects]


# ── 중복 탐지 지문 (완전 중복 해시 + 숫자·변수 마스킹 MinHash) ─────────────────
MINHASH_PERM = 128


def normalize(text: str) -> str:
    t = re.sub(r"\\(dfrac|tfrac)", r"\\frac", text)
    t = re.sub(r"\\displaystyle|\\left|\\right|\\,|\\;|\\!|\\ ", "", t)
    return re.sub(r"\s+", "", t)


def mask(text: str) -> str:
    t = re.sub(r"\d+(\.\d+)?", "<N>", normalize(text))
    return re.sub(r"(?<![A-Za-z\\])[a-zA-Z](?![A-Za-z])", "<V>", t)


def tokens(text: str) -> list[str]:
    return re.findall(r"\\[A-Za-z]+|<[NV]>|[가-힣]+|\S", text)


def minhash(text: str, k: int = 5):
    """마스킹한 LaTeX 토큰 5-gram 의 MinHash 서명(128개, uint64). 두 서명의 일치 비율 ≈ Jaccard 유사도."""
    import numpy as np
    toks = tokens(mask(text))
    sh = {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}
    hv = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh),
                     dtype=np.uint64, count=len(sh))
    rng = np.random.default_rng(20261120)                       # 고정 시드: 빌드마다 같은 서명
    a = rng.integers(1, 2**63 - 1, MINHASH_PERM, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63 - 1, MINHASH_PERM, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return (a[:, None] * hv[None, :] + b[:, None]).min(axis=1)
