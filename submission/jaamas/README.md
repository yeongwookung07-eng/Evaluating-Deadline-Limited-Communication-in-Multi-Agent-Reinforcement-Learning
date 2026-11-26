# JAAMAS 제출 체크리스트 — 2026-11-26

Regular Paper용 LaTeX 제출 자료다. 문서 ZIP을 풀고 각 파일을 투고 시스템의 해당 역할로 지정한다. 이 저장소의 공개는 학술지 제출을 의미하지 않는다.

| 파일 | 제출 역할 | 내용 및 상태 |
|---|---|---|
| [Main_Manuscript.pdf](Main_Manuscript.pdf) | 심사용 본문 | 28쪽, Springer Nature 형식, 초록216단어, 키워드5개 |
| [Manuscript_Source.zip](Manuscript_Source.zip) | LaTeX 원본 | `jaamas_main.tex`를 주 문서로 지정. `.bib`, 그림, 수정 없는 공식 class/style, 부록 및 문서 소스 포함 |
| [Information_Sheet.pdf](Information_Sheet.pdf) | 필수 Information Sheet | 2쪽: Main Claim, Evidence, Related Work, Previous Publications |
| [Cover_Letter.pdf](Cover_Letter.pdf) | 편집자 서신 | 1쪽, 기여·저널 적합성·첨부 자료 |
| [Supplementary.pdf](Supplementary.pdf) | 추가 분석 | 17쪽, S1–S7; 후속 평가와 실패 조건 포함 |
| [Statements_and_Declarations.pdf](Statements_and_Declarations.pdf) | 선언문 사본 | 2쪽 이내. 같은 내용이 본문에도 포함됨 |
| [DECLARATIONS.md](DECLARATIONS.md) | 시스템 입력용 텍스트 | 연구비·이해상충·저자 기여·데이터·코드 공개 |
| [PERMISSIONS.md](PERMISSIONS.md) | 해당 시 권한 확인 | 원저작 그림·표와 외부 코드/데이터 출처 구분 |
| [SHA256SUMS.txt](SHA256SUMS.txt) | 파일 무결성 확인 | 제출 PDF와 소스의 SHA-256 |
| [JAAMAS_Submission.zip](JAAMAS_Submission.zip) | 편의용 묶음 | 위 문서들을 분리 제출하기 위한 ZIP. 원고 소스 대신 이 바깥 ZIP을 올리지 않음 |

Word 원고는 LaTeX 소스의 대체 형식이므로 별도로 만들지 않았다. Highlights와 Graphical Abstract는 활성 제출 묶음에 포함하지 않는다. Reviewer_Archive.zip은 선택 자료이며 저장소 루트에서 `python scripts/package_jaamas.py --skip-build --reviewer`로 생성한다. 큰 ZIP 대신 이 저장소의 코드·데이터·모델·결과를 그대로 이용할 수도 있다.

본문 선언은 저자가 확인한 연구비 없음, 이해상충 없음, 저자 기여를 반영한다. Information Sheet는 이전 학술대회·학술지 출판이 없다는 확인 내용을 사용한다. 공동저자 최종 승인과 중복 투고 여부는 실제 투고 시 제출자가 답한다. 승인·제출 완료를 대신 선언하지 않는다.

공식 근거: [JAAMAS Submission Guidelines](https://link.springer.com/journal/10458/submission-guidelines), 2026-11-26 확인. Information Sheet의 네 질문, 150–250단어 초록, 4–6개 키워드, 편집 가능한 소스와 관련 선언 요건을 확인했다. 투고 시스템에서 요구하는 Author Contributions/Competing Interests도 별도로 입력한다. 모든 포털 필드가 자동으로 검증되었다는 뜻은 아니다.

이번 검토 수정: 본문 5.8의 기본 hard-fallback 초과율 0.100%/0.058%와 적용 범위를 명시했다. 서론·Information Sheet의 식별 결과와 가까운 저널 분석 연구의 관계를 보강했고, Figure 1 글자를 확대했다. 부록 표지에 저널명을 넣고 Methods에 실제 AI 보조 사용 범위를 기재했다. [깨끗한 설치 및 주요 표 검증](../../docs/TABLE_REPRODUCTION.md)은 별도로 실행할 수 있다. LaTeX 원본과 선언문은 준비됐지만 실제 투고 시스템 업로드/입력 여부는 확인하지 않았다.
