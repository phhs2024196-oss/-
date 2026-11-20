-- 논술의 시대 — 문항 DB 스키마 v2 (SQLite 3.35+)
-- 원본은 items/**/*.yaml(문항 1개 = 파일 1개). 이 DB 와 build/items.jsonl 은 tools/build_db.py 가 매번 새로 만드는 파생물.
-- Postgres 로 옮길 때: TEXT(JSON) 열 → JSONB, 나머지는 그대로.

PRAGMA foreign_keys = ON;

CREATE TABLE schools (code TEXT PRIMARY KEY, name TEXT NOT NULL, pilot INTEGER NOT NULL DEFAULT 0,
                      candidate INTEGER NOT NULL DEFAULT 0, format TEXT, scope_2027 TEXT, scope_2028 TEXT);
CREATE TABLE types (school TEXT NOT NULL REFERENCES schools(code), code TEXT NOT NULL, name TEXT NOT NULL,
                    PRIMARY KEY (school, code));

-- ── 문항 본체 ───────────────────────────────────────────────
CREATE TABLE items (
  id TEXT PRIMARY KEY,                      -- SSU-T01-011 / SSU-G26-04 (영구)
  legacy_id TEXT, version INTEGER NOT NULL, schema_version TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('draft','r1_pass','r2_pass','verified','sampled','published','retired')),
  exam_family TEXT NOT NULL, item_type TEXT NOT NULL,
  origin TEXT NOT NULL CHECK (origin IN ('past_exam','mock_exam','original','variant')),
  school TEXT NOT NULL REFERENCES schools(code), type TEXT,
  variant_of TEXT REFERENCES items(id), replaces TEXT REFERENCES items(id),
  level TEXT, form TEXT, points INTEGER NOT NULL, time_estimate_min INTEGER,
  intent TEXT, key_idea TEXT, stuck TEXT, idea_tags TEXT, tags TEXT,
  curriculum_version TEXT, subjects TEXT, scope_2028 TEXT NOT NULL DEFAULT 'unknown', scope_confirmed INTEGER NOT NULL,
  behavior_domain TEXT, unit_path TEXT, key_concepts TEXT, standards TEXT, difficulty TEXT,
  rights_holder TEXT, license TEXT, permission_status TEXT, allowed_use TEXT, derivative_level TEXT, source TEXT,
  verify_ref TEXT, stimuli TEXT, stem TEXT,
  path TEXT NOT NULL, search_text TEXT NOT NULL,
  content_hash TEXT NOT NULL,               -- 교재에 보이는 내용만의 해시(정오표 근거)
  created TEXT NOT NULL, updated TEXT NOT NULL,
  item_json TEXT NOT NULL                   -- 문항 전체(JSONL 한 줄과 동일)
);
CREATE INDEX idx_items_school_type ON items(school, type);
CREATE INDEX idx_items_status ON items(status);

CREATE TABLE subparts (
  item_id TEXT NOT NULL REFERENCES items(id), sid TEXT NOT NULL, ord INTEGER NOT NULL,
  stem TEXT, points INTEGER NOT NULL, depends_on TEXT,
  answer_kind TEXT, answer_value TEXT, answer TEXT,
  PRIMARY KEY (item_id, sid)
);

CREATE TABLE solutions (                   -- 해설 A(방향)·B(전 과정)·ALT(대체 풀이)
  item_id TEXT NOT NULL REFERENCES items(id), solution_id TEXT NOT NULL, sid TEXT NOT NULL,
  audience TEXT NOT NULL, method_no INTEGER, key_idea TEXT, hint_ladder TEXT,
  PRIMARY KEY (item_id, solution_id)
);
CREATE TABLE solution_steps (              -- 해설 단계 ↔ 채점 마크 연결(AI 피드백이 끊긴 단계를 짚는 근거)
  item_id TEXT NOT NULL REFERENCES items(id), solution_id TEXT NOT NULL, step_id TEXT NOT NULL,
  ord INTEGER NOT NULL, content TEXT, criteria TEXT,
  PRIMARY KEY (item_id, solution_id, step_id)
);

CREATE TABLE rubric_criteria (             -- 1마크 = 1행 (AI 첨삭 채점 단위)
  item_id TEXT NOT NULL REFERENCES items(id), criterion_id TEXT NOT NULL, sid TEXT NOT NULL, ord INTEGER NOT NULL,
  mark_type TEXT CHECK (mark_type IS NULL OR mark_type IN ('M','A','B','R','C')),
  points INTEGER NOT NULL, evidence TEXT NOT NULL,
  depends_on TEXT, implied_ok INTEGER, ft TEXT, accept TEXT, cao INTEGER, not_penalized TEXT, steps TEXT,
  PRIMARY KEY (item_id, criterion_id)
);
CREATE TABLE rubric_rules (item_id TEXT NOT NULL REFERENCES items(id), sid TEXT NOT NULL, rules TEXT,
                           PRIMARY KEY (item_id, sid));
CREATE TABLE rubric_deductions (
  item_id TEXT NOT NULL REFERENCES items(id), sid TEXT NOT NULL, code TEXT NOT NULL,
  condition TEXT, effect TEXT CHECK (effect IN ('minus','cap','zero')), value INTEGER,
  PRIMARY KEY (item_id, code)
);

-- ── 오개념·개념·처방 ─────────────────────────────────────────
CREATE TABLE misconceptions (id TEXT PRIMARY KEY, name TEXT NOT NULL, concepts TEXT, retired INTEGER DEFAULT 0);
CREATE TABLE item_misconceptions (
  item_id TEXT NOT NULL REFERENCES items(id), sid TEXT NOT NULL,
  misconception_id TEXT NOT NULL REFERENCES misconceptions(id), outcome_code TEXT NOT NULL,
  wrong_work TEXT, detect_cue TEXT, feedback TEXT, remedy TEXT,
  PRIMARY KEY (item_id, sid, misconception_id)
);
CREATE TABLE concepts (id TEXT PRIMARY KEY, name TEXT NOT NULL, parent TEXT REFERENCES concepts(id), curriculum_code TEXT);
CREATE TABLE concept_edges (src TEXT NOT NULL REFERENCES concepts(id), dst TEXT NOT NULL REFERENCES concepts(id),
                            type TEXT NOT NULL DEFAULT 'precedes', PRIMARY KEY (src, dst, type));
CREATE TABLE remedy_links (
  item_id TEXT NOT NULL REFERENCES items(id), link_id TEXT NOT NULL,
  trigger_type TEXT NOT NULL, trigger_ref TEXT, relation TEXT NOT NULL,
  target_kind TEXT NOT NULL CHECK (target_kind IN ('concept','item')), target_ref TEXT NOT NULL, priority INTEGER,
  PRIMARY KEY (item_id, link_id)
);

-- ── 검수 이력 ─────────────────────────────────────────────
CREATE TABLE reviews (
  item_id TEXT NOT NULL REFERENCES items(id), seq INTEGER NOT NULL,
  stage TEXT NOT NULL, actor_type TEXT, actor_id TEXT, prompt_version TEXT, item_version INTEGER,
  verdict TEXT NOT NULL, reason_codes TEXT, restart_from TEXT, computed_answers TEXT, date TEXT, note TEXT,
  PRIMARY KEY (item_id, seq)
);

-- ── 교재 배치·출고 스냅샷·정오표 ─────────────────────────────
CREATE TABLE books (code TEXT PRIMARY KEY, title TEXT NOT NULL, series TEXT NOT NULL,
                    school TEXT REFERENCES schools(code), season TEXT, edition INTEGER NOT NULL DEFAULT 1);
CREATE TABLE placements (book TEXT NOT NULL REFERENCES books(code), item_id TEXT NOT NULL REFERENCES items(id),
                         chapter TEXT, ord INTEGER NOT NULL, role TEXT, PRIMARY KEY (book, item_id));
CREATE TABLE book_locks (book TEXT NOT NULL REFERENCES books(code), item_id TEXT NOT NULL REFERENCES items(id),
                         version INTEGER NOT NULL, content_hash TEXT NOT NULL, locked TEXT NOT NULL,
                         PRIMARY KEY (book, item_id));
CREATE TABLE errata (                      -- 원본은 git 의 errata/*.yaml
  id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES items(id), reported TEXT NOT NULL, channel TEXT,
  description TEXT NOT NULL, severity TEXT CHECK (severity IN ('typo','wording','answer','rubric','scope')),
  fixed_version INTEGER, status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','confirmed','rejected','fixed'))
);

-- ── 중복 탐지 (3층: 완전 중복 해시 / 숫자·변수 마스킹 MinHash / 발상 태그) ──
CREATE TABLE fingerprints (item_id TEXT PRIMARY KEY REFERENCES items(id), norm_sha256 TEXT NOT NULL,
                           idea_key TEXT, minhash TEXT NOT NULL);
CREATE TABLE dedup_pairs (a TEXT NOT NULL REFERENCES items(id), b TEXT NOT NULL REFERENCES items(id),
                          kind TEXT NOT NULL CHECK (kind IN ('exact','near','idea')), jaccard REAL,
                          PRIMARY KEY (a, b, kind));

-- ── 사이트 단계(2027 여름~)에서 채울 테이블: 지금은 비어 있음 ──
CREATE TABLE grading_runs (               -- AI 첨삭 1회 = 1행. 판독과 채점을 분리해 기록
  run_id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES items(id), item_version INTEGER NOT NULL, sid TEXT NOT NULL,
  image_asset_id TEXT, transcription_latex TEXT, transcription_confirmed INTEGER,
  passes TEXT,                            -- [{criterion_id: points, outcome_code, misconception_ids, evidence_quotes}] × n회
  final_points INTEGER, spread INTEGER, human_review_flag INTEGER, model_id TEXT, prompt_version TEXT, created TEXT
);
CREATE TABLE item_stats (item_id TEXT PRIMARY KEY REFERENCES items(id), attempts INTEGER, p_value REAL,
                         outcome_dist TEXT, updated TEXT);

CREATE VIEW v_type_summary AS
SELECT school, type, COUNT(*) AS n,
       SUM(level='L4') AS l4, SUM(level='L5') AS l5, SUM(level='L6') AS l6, COUNT(DISTINCT form) AS n_forms,
       SUM(status IN ('verified','sampled','published')) AS n_publishable,
       SUM(scope_2028='common') AS n_common, SUM(scope_2028='calc2') AS n_calc2,
       SUM(scope_2028='geom') AS n_geom, SUM(scope_2028='unknown') AS n_unknown
FROM items WHERE status <> 'retired' GROUP BY school, type;

CREATE VIEW v_v2_completeness AS          -- v2 완비율(이관 문항 정리 진도)
SELECT i.school,
       COUNT(DISTINCT i.id) AS items,
       SUM(rc.mark_type IS NOT NULL) * 1.0 / COUNT(rc.criterion_id) AS marks_typed_ratio,
       (SELECT COUNT(*) FROM solutions s JOIN items j ON j.id=s.item_id WHERE j.school=i.school AND s.audience='A') AS solutions_a,
       (SELECT COUNT(*) FROM item_misconceptions m JOIN items j ON j.id=m.item_id WHERE j.school=i.school) AS misconception_links
FROM items i LEFT JOIN rubric_criteria rc ON rc.item_id = i.id GROUP BY i.school;
