# 홍보 영상 지침용 자료 DB 수집 브리프 (2026-11-24)

목적: '논술의 시대'(수리논술 교재·무료 자료 브랜드, 대상 한국 고등학생·학부모) 홍보용 세로 숏폼(릴스·쇼츠) 제작 지침의 근거 DB.
전체 목표 300건 이상, 파트당 80건 이상. 영상 원본은 내려받지 않는다. 링크+분석 기록만.

## 출력: CSV (UTF-8, 헤더 고정, 쉼표 포함 값은 큰따옴표)
id,part,source_type,title,url,org_or_creator,year,domain,format,metric,observation,tags,stance,verified
- id: 파트접두(A/B/C/D)+3자리. 예 A001
- source_type: official(플랫폼 공식 가이드) | research(논문·학술) | stat(조사·통계 보고서) | case(개별 성공 광고·영상 사례) | industry(업계 분석 글)
- domain: edu_kr | edu_global | general
- format: shorts | reels | tiktok | instream_ad | guide | paper | report
- metric: 성과 수치(조회수, 완주율, 브랜드 상기도, 수상 등). 확인 못 하면 빈칸. 절대 지어내지 말 것.
- observation: 한국어 1~3문장. 이 자료가 영상 제작에 대해 말하는 구체적 사실(첫 몇 초에 무엇을, 길이, 자막, 음성, 구성, CTA 등). 숫자는 원문 그대로.
- tags: 아래 고정 태그만, 세미콜론 구분, 1~5개
- stance: support(태그 원칙을 지지) | against(반대 증거) | mixed
- verified: Y(직접 페이지·검색결과로 확인) | P(2차 인용으로만 확인)

## 고정 태그
HOOK_FIRST3S, LENGTH_SHORT(15~30초), LENGTH_MID(30~60초), LENGTH_LONG(60초+), CAPTIONS_TEXT, SOUND_ON_DESIGN, HUMAN_VOICE, TTS_SYNTH, FAST_PACE, PATTERN_INTERRUPT, STORY_ARC, PROBLEM_SOLUTION, DATA_NUMBERS, SPECIFICITY, BRAND_EARLY, BRAND_LATE, CTA_CLEAR, NATIVE_LOOK(광고 티 안 나는 크리에이터식), FACE_PERSON, LOOP_REWATCH, SERIES_FORMAT, SAFE_ZONE, VERTICAL_916, TRENDING_AUDIO, AUTHENTICITY, HOPE_POSITIVE, FEAR_ANXIETY, HUMOR, SOCIAL_PROOF, ONE_MESSAGE, PAYOFF_END, COMMENTS_ENGAGE, POST_FREQUENCY, COVER_THUMBNAIL, DESCRIPTION_LINK, AI_CONTENT_PERCEPTION, VISUAL_DATAVIZ, TEXT_READABILITY, EDU_VALUE_FIRST, PARENT_TARGET, TEEN_TARGET, KOREA_CONTEXT

## 수집 규칙
- 실제로 검색·열람한 자료만 기록. 기억으로 URL이나 수치를 만들지 말 것. 불확실하면 verified=P 또는 metric 빈칸.
- 인스타그램·네이버 블로그·오르비·디시는 robots.txt로 AI 수집 금지 → 페이지를 열지 않는다(검색 결과 요약에 나온 정보만 P로 기록 가능).
- 중복 금지(같은 URL 1번). 다른 파트와 겹쳐도 괜찮지만 파트 주제에 집중.
- 교육·입시 분야를 우선.
