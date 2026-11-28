# JAAMAS 제출 체크리스트 — 2026-11-27

**2026-11-28 현재 투고용 파일.** 초록 220단어, 본문 8pt, 참고문헌·저자 정보 포함 15쪽, 부록 22쪽(S9에 Terran 5v5와 Predator–Prey의 검증 및 시드별 결과 수록). Information Sheet와 Cover Letter에 같은 조건의 짝비교와 적용 범위를 명시했다. 아래 날짜별 수정 기록은 이전 판의 이력이다.

**2026-11-28 최신본:** 3 Problem Definition: 3.1 Problem formulation / 3.2 Communication and receiver state / 3.3 Computation and deadlines. 4절 이후 결과 유지. 저자 정보 포함 전체 20쪽.

**2026-11-28 구성 수정:** 3 Preliminaries / 4 Evaluation framework and experimental design / 5 Results. 총 20쪽; 마지막 페이지에는 참고문헌 끝부분과 저자 정보. 결과·수식·표·그림 보존.

**2026-11-28 최신 수정:** Related Work를 연구 계열·스케줄링 규칙·수신자 역량 평가·우리 분석의 연결로 재작성. 나머지 본문과 실험 결과는 유지. 본문 20쪽.

**2026-11-27 abstract update:** 236-word narrative abstract now follows background → problem → analytical approach → findings → implications. Only the abstract changed; main remains 20 pages. Validation: `narrative_abstract_validation_20261127.json`.

**최신본 (2026-11-27):** 본문 20쪽(참고문헌·선언문·마지막 저자 정보 포함), 부록 19쪽. 첫 페이지는 제목·두 저자 이름·Abstract. 첫 페이지 Highlights/이메일 문구는 삭제. 선행연구 근거 → 분석 질문 → 실제 통제 실험 → 결과와 한계로 재작성. `Title_Page.pdf`는 지침상 필요한 소속·교신저자 연락처를 담은 별도 제출용 표지. 아래 이전 편집 기록의 첫 페이지 설명은 과거 버전임.

**이전 정렬 기록:** 모든 본문·표지 정보·Highlights·캡션은 왼쪽 정렬, 그림·표 자체는 중앙 정렬. 본문 전체 20쪽, 부록 19쪽. 내용과 수치는 그대로이며 정렬과 줄바꿈만 변경했다.

**이전 Highlights 추가 기록:** 첫 페이지 Highlights 5개, 2쪽부터 제목·저자·초록. 본문 전체 20쪽 유지; 부록 18쪽(S1–S8). 상세 진단 그림은 부록으로 이동했고 모든 결과·표·수식·그림은 보존했다.

**이전 20쪽 편집 기록:** 본문 20쪽(참고문헌·선언문 포함), 부록 17쪽(S1–S8). 참고 논문의 문제·기여→방법→실험→결론 흐름에 맞춰 서술을 압축했다. 9개 표와 5개 결과 그림은 본문에, 상세 통제 및 중복 시각화는 부록에 두었다. 전체 표 데이터·수식·그래프 수치는 유지했고 판형·글자 크기·중앙 정렬은 변경하지 않았다.

**이전 판형 변경 기록(2026-11-26):** 첨부 참고 논문의 판형(155 × 235 mm)에 맞춘 본문 31쪽·부록 15쪽. 좌우 여백 16 mm, 본문·그림·표 중앙 배치, 양쪽 정렬, 9.5 pt 본문. 원본 Springer class는 유지하고 `journal-reading-layout.sty`를 추가했다. 과거 버전 기록의 페이지 수는 당시 기준이다.

Regular Paper용 LaTeX 제출 자료다. 문서 ZIP을 풀고 각 파일을 투고 시스템의 해당 역할로 지정한다. 이 저장소의 공개는 학술지 제출을 의미하지 않는다.

| 파일 | 제출 역할 | 내용 및 상태 |
|---|---|---|
| [Main_Manuscript.pdf](Main_Manuscript.pdf) | 심사용 본문 | 15쪽, 본문 8pt, 초록 211단어, 키워드 5개 |
| [Manuscript_Source.zip](Manuscript_Source.zip) | LaTeX 원본 | `jaamas_main.tex`를 주 문서로 지정. `.bib`, 그림, 수정 없는 공식 class/style, 부록 및 문서 소스 포함 |
| [Information_Sheet.pdf](Information_Sheet.pdf) | 필수 Information Sheet | 2쪽: Main Claim, Evidence, Related Work, Previous Publications |
| [Cover_Letter.pdf](Cover_Letter.pdf) | 편집자 서신 | 1쪽, 기여·저널 적합성·첨부 자료 |
| [Supplementary.pdf](Supplementary.pdf) | 추가 분석 | 22쪽, S1–S9. 두 순차 환경의 재생 통제와 학습 시드별 효과 포함 |
| [Statements_and_Declarations.pdf](Statements_and_Declarations.pdf) | 선언문 사본 | 2쪽 이내. 같은 내용이 본문에도 포함됨 |
| [DECLARATIONS.md](DECLARATIONS.md) | 시스템 입력용 텍스트 | 연구비·이해상충·저자 기여·데이터·코드 공개 |
| [PERMISSIONS.md](PERMISSIONS.md) | 해당 시 권한 확인 | 원저작 그림·표와 외부 코드/데이터 출처 구분 |
| [SHA256SUMS.txt](SHA256SUMS.txt) | 파일 무결성 확인 | 제출 PDF와 소스의 SHA-256 |
| [JAAMAS_Submission.zip](JAAMAS_Submission.zip) | 편의용 묶음 | 위 문서들을 분리 제출하기 위한 ZIP. 원고 소스 대신 이 바깥 ZIP을 올리지 않음 |

Word 원고는 LaTeX 소스의 대체 형식이므로 별도로 만들지 않았다. Highlights와 Graphical Abstract는 활성 제출 묶음에 포함하지 않는다. Reviewer_Archive.zip은 선택 자료이며 저장소 루트에서 `python scripts/package_jaamas.py --skip-build --reviewer`로 생성한다. 큰 ZIP 대신 이 저장소의 코드·데이터·모델·결과를 그대로 이용할 수도 있다.

본문 선언은 저자가 확인한 연구비 없음, 이해상충 없음, 저자 기여를 반영한다. Information Sheet는 이전 학술대회·학술지 출판이 없다는 확인 내용을 사용한다. 공동저자 최종 승인과 중복 투고 여부는 실제 투고 시 제출자가 답한다. 승인·제출 완료를 대신 선언하지 않는다.

공식 근거: [JAAMAS Submission Guidelines](https://link.springer.com/journal/10458/submission-guidelines), 2026-11-26 확인. Information Sheet의 네 질문, 150–250단어 초록, 4–6개 키워드, 편집 가능한 소스와 관련 선언 요건을 확인했다. 투고 시스템에서 요구하는 Author Contributions/Competing Interests도 별도로 입력한다. 모든 포털 필드가 자동으로 검증되었다는 뜻은 아니다.

이전 검토 기록(후속 수정 전): 본문 5.8의 기본 hard-fallback 초과율 0.100%/0.058%와 적용 범위를 명시했다. 서론·Information Sheet의 식별 결과와 가까운 저널 분석 연구의 관계를 보강했고, Figure 1 글자를 확대했다. 부록 표지에 저널명을 넣고 Methods에 실제 AI 보조 사용 범위를 기재했다. [깨끗한 설치 및 주요 표 검증](../../docs/TABLE_REPRODUCTION.md)은 별도로 실행할 수 있다. LaTeX 원본과 선언문은 준비됐지만 실제 투고 시스템 업로드/입력 여부는 확인하지 않았다.

현재 재작성본은 KBS와 실제 수행 기록을 근거로 “왜 필요한 연구인가 → 무엇을 구분하려는가 → 왜 이 통제 비교가 필요한가 → 결과로 무엇까지 설명되는가” 순서로 구성했다. 3절에서 문제 정의와 분석 논리를 먼저 설명하고, 기존 결과 수치와 한계는 유지했다. Information Sheet와 Cover Letter도 같은 논리에 맞췄다. [근거 대응표](../../paper/problem_rationale_evidence_map.json).

이전 발견 중심 수정 기록(후속 도식 삭제 전, 2026-11-26): 초록·서론·Discussion·Conclusion 재작성, 절 제목과 실험 간 연결 정리, 기존 결과 기반 그림 6개 추가(본문 총 8개). 기존 표 11개, 수식, 부록 원문과 참고문헌 유지. 현재 본문 33쪽. AI-use 문단 삭제는 직전 요청대로 유지.

2026-11-26: 설명용 평가 절차 도식 삭제. 설명은 본문에 보존하고 실제 실험 결과 그래프만 유지. 본문 33쪽; 표·수식·실험 결과 변경 없음.

이전 교정본 기록(판형 변경 전, 2026-11-26): 본문 33쪽, 부록 17쪽, 본문 실험 그래프 7개. 오참조 두 곳과 표현을 교정하고 본문·별도 PDF·DECLARATIONS.md의 선언문을 통일했다. 과거 기록의 그림 수와 AI-use 서술은 현행 상태를 나타내지 않는다. 실험 결과·표 데이터·수식·참고문헌 항목은 유지했다.
