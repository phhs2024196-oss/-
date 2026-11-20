"""논술의 시대 문항 DB 공용 함수: 문항 md 파싱, 분류표 로딩, 출제범위 추정."""
from __future__ import annotations

import hashlib
import pathlib
import re

import yaml

KIT = pathlib.Path(__file__).resolve().parent.parent          # nonsul-era/
TAXONOMY_PATH = KIT / "db" / "taxonomy.yaml"

SECTIONS = ["제시문", "논제", "모범답안", "정답", "채점표", "대표 오답", "메모"]
REQUIRED_SECTIONS = ["논제", "모범답안", "정답", "채점표"]
STATUS_ORDER = ["draft", "r1_pass", "r2_pass", "verified", "sampled", "published"]

ID_ORIGINAL = re.compile(r"^(?P<school>[A-Z]{2,4})-(?P<type>T\d{2})-(?P<seq>\d{3})$")
ID_PAST = re.compile(r"^(?P<school>[A-Z]{2,4})-(?P<src>[GM])(?P<yy>\d{2})(?P<session>[A-Z]?)-(?P<no>\d{2})$")
STEP_MARKS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"


def load_taxonomy() -> dict:
    return yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8"))


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise ValueError("front matter(--- ... ---) 없음")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("front matter 닫는 --- 없음")
    meta = yaml.safe_load(text[4:end]) or {}
    return meta, text[end + 5:]


def split_sections(body: str) -> dict[str, str]:
    """'## 제목' 단위로 본문을 나눈다. 알 수 없는 제목도 그대로 보관."""
    out: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m and not line.startswith("###"):
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def parse_rubric(table_md: str) -> tuple[list[dict], list[str]]:
    """채점표 표 → ([{seq, step, sub_question, content, points}], [읽지 못한 행]).
    3열(단계|내용|배점) 과 4열(단계|소문항|내용|배점) 모두 지원.
    머리행은 마지막 칸이 정확히 '배점'인 행. 배점 칸은 정수만 허용(14.5 같은 값은 오류로 보고)."""
    rows, bad = [], []
    for line in table_md.splitlines():
        s = line.strip()
        if not s.startswith("|") or re.match(r"^\|[\s:|-]*$", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and cells[-1] == "배점":
            continue
        if len(cells) < 3 or not re.fullmatch(r"\d+", cells[-1]):
            bad.append(s)
            continue
        if len(cells) >= 4:
            step, sub, content = cells[0], cells[1], " | ".join(cells[2:-1])
        else:
            step, sub, content = cells[0], None, cells[1]
        if sub in ("-", "", "—"):
            sub = None
        rows.append({"seq": len(rows) + 1, "step": step, "sub_question": sub,
                     "content": content, "points": int(cells[-1])})
    return rows, bad


CONTENT_FIELDS = ("answer", "points", "sub_points", "level", "key_idea")


def item_content_hash(meta: dict, body: str) -> str:
    """교재에 보이는 내용만 해시: 본문 + 정답·배점·소문항 배점·난이도·열쇠.
    status·reviews·updated 같은 관리 필드 변경은 정오표 대상이 아니므로 제외."""
    import json
    visible = {k: meta.get(k) for k in CONTENT_FIELDS}
    return content_hash(json.dumps(visible, ensure_ascii=False, sort_keys=True, default=str) + "\n" + body)


def content_hash(text: str) -> str:
    """문자열 sha256 앞 16자."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_item(path: pathlib.Path) -> tuple[dict, dict, str]:
    text = path.read_text(encoding="utf-8")
    meta, body = split_front_matter(text)
    return meta, split_sections(body), body


def item_seq(item_id: str) -> int | None:
    m = ID_ORIGINAL.match(item_id or "")
    return int(m["seq"]) if m else None


def all_item_ids() -> set[str]:
    """저장소 전체 문항 ID(파일명 기준) — 일부 폴더만 검사할 때 parent·교재 참조 확인용."""
    return {p.stem for p in (KIT / "items").rglob("*.md")}


def load_books() -> list[tuple[pathlib.Path, dict]]:
    out = []
    for bf in sorted((KIT / "books").glob("*.yaml")):
        if bf.name.endswith(".lock.yaml"):
            continue
        out.append((bf, yaml.safe_load(bf.read_text(encoding="utf-8")) or {}))
    return out


# ── 2028 출제범위 추정 (키워드 휴리스틱: 확정이 아니라 태깅 우선순위용) ──────────
GEOM_PAT = re.compile(r"\\overrightarrow|벡터|내적|타원|쌍곡선|준선|정사영|이면각|삼수선|좌표공간|공간좌표")
TRANS_PAT = re.compile(r"\\ln|\\log|e\^|\\sin|\\cos|\\tan|\\sec|\\csc|\\cot")
CALC_PAT = re.compile(r"f'|\\int|미분|적분|접선|극값|극대|극소|도함수|넓이|부피")
CALC2_KW = re.compile(
    r"치환적분|부분적분|역함수의 미분|음함수|매개변수|덧셈정리|이계도함수|변곡점|곡선의 길이|"
    r"급수|\\lim_\{n\s*\\to\s*\\infty\}|\\sum_\{[a-z]=1\}\^\{?\\infty|속도|가속도|구분구적"
)
# 확률과통계 고유 도구(고1 '경우의 수·순열·조합'은 제외) — exclude_tools 검사용
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
    if "기하" in subjects:
        scope = "geom"
    elif "미적분" in subjects:
        scope = "calc2"
    else:
        scope = "common"
    order = ["수학", "수학Ⅰ", "수학Ⅱ", "미적분", "확률과통계", "기하"]
    return scope, [s for s in order if s in subjects]
