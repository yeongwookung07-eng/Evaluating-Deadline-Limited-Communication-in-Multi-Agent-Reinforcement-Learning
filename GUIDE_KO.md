# GitHub 공개용 연구 구성 안내

이 폴더의 **내용 전체를 GitHub 저장소 루트에 올리면 됩니다**. 원본 연구 저장소에서 현재 KBS 원고에 관련된 코드, 설정, 테스트, 작은 결과 파일과 논문을 추린 독립 복사본입니다. 별도 분할 ZIP이나 업로드 도구는 필요하지 않습니다.

## 연구 질문

행동 마감시간과 수신 캐시가 있는 다중 에이전트 강화학습에서 통신량 감소의 원인을 구분합니다. 전송 시도 감소, 제때 도착한 메시지, 발신자 선택, 초기화, CPU 계산시간, 정책의 정보 활용을 각각 검증합니다. DAC는 평가 대상 휴리스틱이며 보편적으로 우수한 새 학습 알고리즘이라는 주장이 아닙니다.

## 먼저 읽을 파일

- [README](README.md): 연구 개요와 핵심 결과.
- [논문 PDF](experiments/deadline_comm/publication/kbs/kbs_main.pdf): 현재 연구의 전체 설명. 같은 폴더에 영문 LaTeX와 참고문헌, 그림이 있습니다.
- [근거 지도](EVIDENCE.md): 논문의 주장에 대응하는 코드와 결과 디렉터리.
- [검증 및 실행](REPRODUCE.md): 설치, CPU 테스트, 추가 입력이 필요한 분석 명령.
- [출처](THIRD_PARTY_NOTICES.md): QMIX, Hallway, SMACv2와 논문 템플릿의 출처 및 이용 조건.

## 중요한 구현

아래 경로는 별도 표기가 없으면 `experiments/deadline_comm/` 아래에 있습니다.

| 구성 | 역할과 입출력 |
| --- | --- |
| `transport.py`, `timing.py` | 통신 예산·메시지·계산 준비시점에 따른 전송/수신 처리. 시도한 wire bytes와 도착한 payload, 마감시간을 구분합니다. |
| `harness.py`, `policy.py`, `replay.py` | 정책 실행, 수신 캐시, 재생을 연결합니다. 같은 체크포인트와 초기 상태를 유지하는 비교의 기반입니다. |
| `configs/` | 각 연구의 시드, 예산, 타이밍, 체크포인트와 출력 경로. 기존 결과를 덮어쓰지 않도록 실행 전 확인합니다. |
| `review_revision/identity.py` | Full–RR의 수신 기록·캐시·행동·보상 일치를 검사합니다. |
| `tmlr_revision/experiments.py` | 동일 수신 개수에서 발신자를 바꾸는 개입과 공통 RR 초기 구간 비교. sender 효과와 초기화 효과를 분리합니다. |
| `tmlr_revision/readiness.py`, `tmlr_revision/robustness/cpu_repeat.py` | 선택 계산시간과 고정/측정 readiness의 영향을 분석합니다. |
| `tmlr_revision/broadcast.py`, `tmlr_revision/robustness/aggregate_probe.py` | Oracle-free broadcast 및 실제 수신 입력 probe. 정보가 존재하는 것과 정책이 활용하는 것을 구분합니다. |
| `tmlr_revision/local_clock.py` | Hallway에서 통신 없이 공개 시계와 로컬 거리로 해결하는 반례입니다. |
| `reviewer_validation/`, `standard_benchmarks/`, `smac_protocol/` | 학습 gate, 실패 결과 및 SMACv2 기술 smoke. 표준 벤치마크 학습 성공으로 해석하지 않습니다. |
| `tmlr_revision/results/`, `runs/` | 보존된 집계·시드별 작은 결과. 새 실행으로 얻은 값이 아닙니다. |
| 루트 `models/qmix/`, `framework/` | 기존 QMIX와 에이전트·버퍼·학습기 의존 코드입니다. 새 제안 모델이 아닙니다. |
| 루트 `external/` | 필요한 Hallway 원본과 라이선스. 전체 외부 벤치마크 설치본은 아닙니다. |

## 해석할 때 유지해야 하는 경계

Oracle-assisted Hallway와 Oracle-free broadcast는 별도 근거입니다. 400개 Full–RR 쌍의 수신 동일성과 80% 전송 시도 감소는 명시된 고정 readiness 조건의 결과입니다. 발신자 효과는 상태/에피소드 가중 방식에 민감하며 DAC의 일반적 우위를 확립하지 않습니다. probe 정확도는 실제 정책의 달성 성능이 아닙니다. 무통신 local-clock 해법, XOR/SMACv2 학습 gate 실패, SMACv2 기술 smoke의 제한도 함께 보존했습니다.

## 포함하지 않은 것

대용량 체크포인트, NPZ 입력, 상세 단계별 trace, 가상환경, 학습 로그 전체, 과거의 다른 연구, 분할 업로드 ZIP은 제외했습니다. 원본 작업 공간에는 그대로 남아 있습니다. 따라서 이 폴더만으로 코드 테스트와 작은 결과 열람은 가능하지만, 모든 논문 실험을 다시 실행할 수 있는 완전한 데이터 배포본은 아닙니다. 이전 문서의 역사적 실행 경로나 생략된 artifact 링크는 원본 작업 공간을 가리킬 수 있습니다.

이 작업은 파일 정리이며 신규 학습·실험 또는 GitHub 업로드를 수행하지 않았습니다.
