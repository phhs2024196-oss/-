-- 논술의 시대 — 문항 DB 스키마 (SQLite 3.35+; Postgres 이전 시 TEXT JSON → JSONB 로만 바꾸면 됨)
-- 원본(source of truth)은 items/**/*.md 이고, 이 DB 는 tools/build_db.py 가 매번 새로 만드는 파생물이다.
-- 따라서 DB 를 직접 수정하지 않는다. 수정은 항상 md → build.

PRAGMA foreign_keys = ON;

CREATE TABLE schools (
  code          TEXT PRIMARY KEY,            -- SSU, DGU, SJU, COM ...
  name          TEXT NOT NULL,
  pilot         INTEGER NOT NULL DEFAULT 0,
  candidate     INTEGER NOT NULL DEFAULT 0,
  format        TEXT,
  scope_2027    TEXT,                        -- JSON 배열
  scope_2028    TEXT                         -- JSON 배열 또는 'unknown'
);

CREATE TABLE types (
  school        TEXT NOT NULL REFERENCES schools(code),
  code          TEXT NOT NULL,               -- T01 ...
  name          TEXT NOT NULL,
  PRIMARY KEY (school, code)
);

CREATE TABLE items (
  id            TEXT PRIMARY KEY,            -- SSU-T01-001 / SSU-G26-04 / SJU-G26A-01
  legacy_id     TEXT,                        -- S01-01 (기존 300문항)
  kind          TEXT NOT NULL CHECK (kind IN ('past','mock','original','variant')),
  school        TEXT NOT NULL REFERENCES schools(code),
  type          TEXT,                        -- T01 ...
  parent        TEXT REFERENCES items(id),   -- variant 의 원 문항
  level         TEXT CHECK (level IN ('L3','L4','L5','L6')),
  form          TEXT,
  points        INTEGER NOT NULL,
  sub_points    TEXT,                        -- JSON 배열 [8,10,12]
  subjects      TEXT,                        -- JSON 배열 (2015 개정 과목명)
  scope_2028    TEXT NOT NULL DEFAULT 'unknown' CHECK (scope_2028 IN ('common','calc2','geom','unknown')),
  scope_confirmed INTEGER NOT NULL DEFAULT 0, -- 0 = 휴리스틱 추정값, 1 = 사람/검토 모델이 확인
  stuck         TEXT,                        -- 학생이 막히는 지점(AI 첨삭 힌트 근거)
  concepts      TEXT,                        -- JSON 배열
  key_idea      TEXT,                        -- 열쇠 한 줄
  answer        TEXT,                        -- 정답(표시용 LaTeX)
  status        TEXT NOT NULL CHECK (status IN ('draft','r1_pass','r2_pass','verified','sampled','published','retired')),
  verify_ref    TEXT,                        -- verify/SSU-T01.py::p01
  source        TEXT,                        -- JSON {school, year, session, number, url}
  original_text INTEGER NOT NULL DEFAULT 0,  -- 기출 원문 수록 허락 여부 (0 = 출처 ID + 자체 해설만)
  version       INTEGER NOT NULL DEFAULT 1,
  created       TEXT NOT NULL,
  updated       TEXT NOT NULL,
  path          TEXT NOT NULL,               -- items/SSU/T01/SSU-T01-001.md
  body_md       TEXT NOT NULL,               -- 본문 전체(제시문~해설)
  passage_md    TEXT, prompt_md TEXT, solution_md TEXT, rubric_md TEXT, errors_md TEXT,
  content_hash  TEXT NOT NULL                -- 본문 sha256 앞 16자: 수정 감지·정오표 근거
);
CREATE INDEX idx_items_school_type ON items(school, type);
CREATE INDEX idx_items_status ON items(status);
CREATE INDEX idx_items_scope ON items(scope_2028);

CREATE TABLE rubric_steps (               -- 채점표 한 줄 = 한 행 (AI 첨삭이 이 단위로 채점)
  item_id       TEXT NOT NULL REFERENCES items(id),
  step          TEXT NOT NULL,              -- ①, ② ...
  sub_question  TEXT,                       -- (1)(2)(3) — 다소문항일 때
  content       TEXT NOT NULL,
  points        INTEGER NOT NULL,
  PRIMARY KEY (item_id, step)
);

CREATE TABLE reviews (                    -- 검토 이력 (front matter reviews[])
  item_id       TEXT NOT NULL REFERENCES items(id),
  seq           INTEGER NOT NULL,
  stage         TEXT NOT NULL,              -- r1 | r2a | r2b | verify | sample | errata
  reviewer      TEXT NOT NULL,              -- 모델명 또는 'human'
  date          TEXT NOT NULL,
  result        TEXT NOT NULL CHECK (result IN ('pass','fail','fixed')),
  note          TEXT,
  PRIMARY KEY (item_id, seq)
);

CREATE TABLE books (                      -- books/*.yaml
  code          TEXT PRIMARY KEY,           -- SSU-YUYU-2028 (숭실 유유상종 2028 시즌)
  title         TEXT NOT NULL,
  series        TEXT NOT NULL,              -- 기생기사·다다익선·백전불태·유유상종·난공불락·만류기종·속전속결 ...
  school        TEXT REFERENCES schools(code),
  season        TEXT,                       -- 2028
  edition       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE placements (                 -- 한 문항이 여러 교재에 실릴 수 있음
  book          TEXT NOT NULL REFERENCES books(code),
  item_id       TEXT NOT NULL REFERENCES items(id),
  chapter       TEXT,
  ord           INTEGER NOT NULL,
  PRIMARY KEY (book, item_id)
);

CREATE TABLE errata (                     -- 정오표 / 오류 신고
  id            INTEGER PRIMARY KEY,
  item_id       TEXT NOT NULL REFERENCES items(id),
  reported      TEXT NOT NULL,
  reporter      TEXT,                       -- 개인정보 저장 금지: 채널명만 (스토어 문의, 메일)
  description   TEXT NOT NULL,
  severity      TEXT CHECK (severity IN ('typo','wording','answer','rubric','scope')),
  fixed_version INTEGER,
  status        TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','confirmed','rejected','fixed'))
);

-- 전문 검색 (문항 본문·열쇠·개념)
CREATE VIRTUAL TABLE items_fts USING fts5(id UNINDEXED, body_md, key_idea, concepts, tokenize = 'unicode61');

-- 자주 쓰는 집계 뷰
CREATE VIEW v_type_summary AS
SELECT school, type,
       COUNT(*)                                        AS n,
       SUM(level = 'L4') AS l4, SUM(level = 'L5') AS l5, SUM(level = 'L6') AS l6,
       COUNT(DISTINCT form)                            AS n_forms,
       SUM(status IN ('verified','sampled','published')) AS n_publishable,
       SUM(scope_2028 = 'common') AS n_common, SUM(scope_2028 = 'calc2') AS n_calc2,
       SUM(scope_2028 = 'geom') AS n_geom,   SUM(scope_2028 = 'unknown') AS n_unknown
FROM items WHERE status <> 'retired'
GROUP BY school, type;
