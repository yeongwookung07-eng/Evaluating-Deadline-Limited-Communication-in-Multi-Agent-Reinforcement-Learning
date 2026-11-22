"""Read-only analysis of retained day-one/day-two artifacts; no training."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.deadline_comm.day2.aggregate_day2 import read_main
from experiments.deadline_comm.day2.close_day2_gaps import timing_statistics, PHASES

BASE=Path('experiments/deadline_comm/runs')
OUT=BASE/'day3_final_analysis_20261118_001'
OUT.mkdir(exist_ok=True)
main=BASE/'day2_dac_marl_main_20261118_001'
follow=BASE/'day2_followup_20261118_001'
gap=BASE/'day2_gap_closure_20261118_001'
sources=[]

def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

records,convergence=read_main(main)
for f in sorted(follow.glob('seed*/evaluation.json')):
    records.extend(json.loads(f.read_text()));sources.append(f)
records.extend(json.loads((gap/'evaluation.json').read_text()))
write_csv('day2_seed_metrics.csv',records)
index={(r['seed'],r['condition'],r['method']):r for r in records if r['category']!='ablation'}
deltas=[]
for condition in ['slow_5ms','medium_2ms','medium_1.5ms']:
    for seed in range(3):
        a=index[seed,condition,'dac_comm_aware']['mean_return']
        b=index[seed,condition,'comm_aware_lambda1']['mean_return']
        c=index[seed,condition,'dac_on_2k']['mean_return']
        deltas.append(dict(condition=condition,seed=seed,comm_aware=a,no_decay=b,
                           decay_delta=a-b,full_2k_dac=c,adaptation_delta=a-c))
write_csv('paired_seed_differences.csv',deltas)
summary=[]
groups=defaultdict(list)
for r in records:
    groups[r['category'],r['condition'],r['method']].append(r)
for (category,condition,method),rows in sorted(groups.items()):
    for cohort,selected in [('primary_all_seeds',rows),
                            ('secondary_return_ge5',[r for r in rows if convergence[r['seed']]>=5])]:
        row=dict(cohort=cohort,category=category,condition=condition,method=method,
                 seeds=','.join(str(r['seed']) for r in selected),
                 all_budget_eligible=all(r['budget_eligible'] for r in selected))
        for metric in ['mean_return','offered_B_step','goodput_payload_B_step','wasted_pct','cache_invalid_pct','drop_pct']:
            values=[r[metric] for r in selected if r[metric] is not None]
            row[metric+'_mean']=float(np.mean(values)) if values else None
            row[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(row)
write_csv('primary_secondary_tables.csv',summary)

# Parse raw timing, independently verify timer identities and retained statistics.
audit=[]
timing=[]
mismatches=[]
saved=json.loads((gap/'final_tables.json').read_text())['timings']
saved_index={(r['run'],r['seed'],r['condition'],r['method']):r for r in saved}
for root in [main,follow,gap]:
    for path in sorted(root.glob('seed*/*/steps.csv')):
        sources.append(path)
        with path.open() as f:rows=list(csv.DictReader(f))
        grouped=defaultdict(list)
        for row in rows:grouped[row.get('arm',row['method'])].append(row)
        for method,ss in grouped.items():
            seed=int(path.parts[-3][4:])
            values={k:np.array([float(r[k]) for r in ss]) for k in PHASES}
            parts=sum(values[k] for k in PHASES[:5])
            identity=np.max(np.abs(parts-values['cpu_total_ns']))
            total_error=max(abs(float(r['step_total_ns'])-float(r['cpu_total_ns'])-float(r['comm_model_ns'])) for r in ss)
            nonfinite=sum(int((~np.isfinite(v)).sum()) for v in values.values())
            negative=sum(int((v<0).sum()) for v in values.values())
            stats=timing_statistics(ss)
            key=(root.name,seed,path.parent.name,method)
            for phase in PHASES:
                for stat in ['mean','median','p95']:
                    if abs(stats[phase][stat]-saved_index[key]['timings_us'][phase][stat])>1e-7:
                        mismatches.append(str(key))
                timing.append(dict(run=root.name,seed=seed,condition=path.parent.name,method=method,
                                   phase=phase,**stats[phase]))
            audit.append(dict(run=root.name,seed=seed,condition=path.parent.name,method=method,
                              steps=len(ss),phase_sum_error_ns=float(identity),step_identity_error_ns=float(total_error),
                              nonfinite=nonfinite,negative=negative,
                              cpu_max_ms=float(values['cpu_total_ns'].max()/1e6),
                              cpu_over_5ms=int((values['cpu_total_ns']>5e6).sum()),
                              action_misses=sum(r['action_miss'].lower()=='true' for r in ss)))
write_csv('cpu_phase_statistics_us.csv',timing)
write_csv('timing_audit.csv',audit)
assert not mismatches and all(r['phase_sum_error_ns']<1e-4 and r['step_identity_error_ns']<1e-4 and not r['negative'] and not r['nonfinite'] for r in audit)

# Historical single-seed evidence is a separate column, never pooled with day two.
historical=BASE/'bc64_preliminary_001/summary.json'
old=json.loads(historical.read_text())['groups']
integration=[]
for r in old:
    matches=[x for x in summary if x['cohort']=='primary_all_seeds' and x['category']=='main'
             and x['condition']==r['condition'] and x['method']==r['method']]
    if not matches:continue
    x=matches[0]
    integration.append(dict(condition=r['condition'],method=r['method'],day1_training_seeds=r['training_seeds'],
         day1_return=r['mean_return'],day2_return_mean=x['mean_return_mean'],day2_return_sd=x['mean_return_sd'],
         day1_offered_B=r['mean_wire_bytes'],day2_offered_B=x['offered_B_step_mean'],
         day1_drop_pct=100*r['message_drop_rate'],day2_drop_pct=x['drop_pct_mean']))
write_csv('day1_day2_comparison.csv',integration)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
conditions=[f'{link}_{d}ms' for link in ['fast','medium','slow'] for d in ['5','2.5','2','1.5']]
system_methods=['full','periodic','vbc_batched','tmc','dac']
colors=['#777777','#0072B2','#E69F00','#CC79A7','#009E73']
def stats(condition,method,metric):
    values=[index[s,condition,method][metric] for s in range(3)]
    values=[v for v in values if v is not None]
    return (np.mean(values),np.std(values,ddof=1)) if len(values)>1 else (np.nan,np.nan)
fig,axes=plt.subplots(3,3,figsize=(16,11),sharex='col')
for col,link in enumerate(['fast','medium','slow']):
    for row,metric in enumerate(['offered_B_step','goodput_payload_B_step','wasted_pct']):
        ax=axes[row,col]
        for method,color in zip(system_methods,colors):
            ys=[stats(f'{link}_{d}ms',method,metric) for d in ['5','2.5','2','1.5']]
            ax.errorbar(range(4),[v[0] for v in ys],yerr=[v[1] for v in ys],label=method,color=color,marker='o',capsize=3)
        ax.set_ylim([(0,260),(0,175),(-2,105)][row])
        if row==0:ax.axhspan(114,126,color='grey',alpha=.12);ax.set_title(link)
        if col==0:ax.set_ylabel(['Offered load (B/step)','Payload goodput (B/step)','Wasted bandwidth (%)'][row])
        ax.set_xticks(range(4),['5','2.5','2','1.5']);ax.grid(alpha=.2)
        if row==2:ax.set_xlabel('Deadline (ms), decreasing')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=5)
fig.suptitle('System efficiency — base checkpoints, 3-seed mean ± sample SD',y=.97)
fig.tight_layout(rect=[0,0,1,.94])
for ext in ['png','pdf']:fig.savefig(OUT/f'figure1_system_efficiency.{ext}')
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
methods=['dac_on_2k','dac_comm_aware','periodic','vbc','tmc']
labels=['Full-Comm 2k + DAC','Comm-Aware 2k','Periodic (1.5k)','VBC scalar (1.5k)','TMC (1.5k)']
for ax,link in zip(axes,['fast','medium','slow']):
    for method,label,color in zip(methods,labels,colors):
        ys=[stats(f'{link}_{d}ms',method,'mean_return') for d in ['5','2.5','2','1.5']]
        ax.errorbar(range(4),[v[0] for v in ys],yerr=[v[1] for v in ys],label=label,color=color,marker='o',capsize=3)
    ax.set_title(link);ax.set_xticks(range(4),['5','2.5','2','1.5'])
    ax.set_xlabel('Deadline (ms), decreasing');ax.grid(alpha=.2)
axes[0].set_ylabel('Return')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3)
fig.tight_layout(rect=[0,0,1,.84])
for ext in ['png','pdf']:fig.savefig(OUT/f'figure2_control_tradeoffs.{ext}')
plt.close(fig)

lines=['# 실시간 마감·정보 노후화 MARL 통신 실험 최종 보고서 초안','',
'## 연구 질문과 측정 계약','',
'마감 내 메시지 수신을 제한하고 이전 캐시로 행동할 때, 송신량·유효 수신량·제어 성능이 어떻게 달라지는지 Hallway에서 검증했다. 기존 VBC/TMC 논문의 native 재현이 아닌 로컬 스타일 게이트 비교다.',
'CPU 실측 연산 시간에 직렬 공유 링크의 모델 대기시간을 더했다. R_post=0.45ms이며 수신 cutoff 이후 메시지는 폐기한다. 실제 무선망의 측정 결과나 하드 실시간 보장이 아니다.',
'48B 패킷은 payload 32B와 헤더 16B로 구성한다. Goodput은 수신 성공 broadcast payload를 한 번만 센다. 캐시 미수신율은 무효 캐시 비율이며 유효하지만 오래된 캐시까지 포함하지 않는다.','',
'## 분석 기준','',
'주 결과는 학습 seed 0,1,2 전체 평균과 표본 표준편차이다. 평가 reset 20개를 세 번 반복한 것은 CPU 변동 관측이며 독립 환경 seed 60개로 해석하지 않는다.',
'Full-Comm 최종 평가 Return은 9.33,4.00,6.00이다. Return>=5 조건부 보조 표는 seed0,2만 포함하며 선택 편향을 제거하는 장치는 아니다.',
'1일차 단일 seed와 2일차 3-seed는 day1_day2_comparison.csv에서 병기하며 합쳐 평균내지 않는다. 초기 Return=0 진단 실험을 학습된 모델 성능과 혼합하지 않는다.','',
'## 시스템 효율','',
'관측된 slow 5ms 조건에서 Full/Periodic/DAC는 각각 240/120/48 B/step을 보내고 모두 payload goodput 32 B/step을 얻었다. 폐기율은 각각 80/60/0%였다. 이는 이 링크 모델과 조건에서 혼잡 상한의 전송 낭비 감소를 보여준다.',
'폐기율 0%는 모든 조건의 보장이 아니다. 실측 연산 변동이 송신 선택 이후 발생하면 드롭될 수 있다. slow 1.5ms의 DAC는 송신 0이므로 폐기율은 정의되지 않으며 0%로 표시하지 않는다.',
'120 B/step은 평균 예산이다. DAC 상한은 누적 예산을 제한하며 병목 링크에서 114B 하한을 만족하지 못한다. 정보 공급 부족은 드롭·무효 캐시·노후도와 Return을 함께 해석하며 별도 인과 조작 없이 단일 원인을 확정하지 않는다.',
'![System efficiency](figure1_system_efficiency.png)','',
'## 동일 학습량 제어 성능','',
'Full-Comm +500과 Comm-Aware +500은 optimizer/target 재초기화, lr=1e-4, Oracle4+Online12, BC=.5, fixed-horizon Double Q를 맞췄다. 온라인 궤적과 Oracle 수신 이력은 학습 조건에 따라 달라진다.',
'medium 5ms에서 Full-Comm 2k+DAC는 12.83±5.06으로 Comm-Aware 8.50±4.36보다 높았다. slow5ms와 medium1.5ms에서는 Comm-Aware 평균이 높았다. 통신 조건 적응의 효과는 조건 의존적이며 추가 학습의 효과도 크다.',
'Figure2의 Periodic/VBC/TMC는 1500-update 기본 모델이다. 2000-update 두 학습군과의 비교에는 학습량 차이가 있으므로 순수 알고리즘 우위로 읽지 않는다.',
'![Control tradeoffs](figure2_control_tradeoffs.png)','',
'## 감쇠의 seed별 차이','',
'차이는 같은 seed의 Comm-Aware Return에서 평가 시 λ=1 Return을 뺀 값이다.','',
'| 조건 | Seed | 감쇠 사용 | 감쇠 제거 | 차이 |',
'|---|---|---|---|---|']
for r in deltas:lines.append(f"| {r['condition']} | {r['seed']} | {r['comm_aware']:.3f} | {r['no_decay']:.3f} | {r['decay_delta']:+.3f} |")
lines+=['','slow5ms는 +1.00,+1.50,-0.50이고 medium2ms는 0,+1.00,-0.167, medium1.5ms는 0,+1.333,-0.333이다. 전 seed에서 일관된 이득은 아니다. medium 두 조건의 양의 평균은 seed1이 주도하고 seed2에서는 역효과였다.',
'본 조건에서는 감쇠의 일관된 추가 이득이 관찰되지 않았다. 단순화 후보로 λ=1을 유지하되, λ=1로 학습한 모델까지 검증한 것은 아니다. 기존 체크포인트와 결과를 보존한다.','',
'## 분산형 결과','',
'보정 이후 medium5ms에서 점수 기반·Slot-Only 모두 120 B/step과 Return1.667±1.528을 보였다. 병목 조건에서도 관측 Return이 같았다. 전송 확률이 포화되어 같은 슬롯을 쓰는 경우에는 점수의 효용을 식별할 수 없다. 점수의 추가 이득 미관찰과 보편적 불필요성을 구분한다.',
'분산 실행은 공통 step/ID/N 및 고정된 연산 시간 여유를 가정한다. 중앙 Top-K는 비교 기준이며 이론적 최적 성능 상한이 아니다.','',
'## CPU 타이밍 감사','']
lines += [f"원시 로그 {sum(r['steps'] for r in audit):,}스텝, {len(audit)}개 seed·조건·방법 그룹을 재계산했다. 저장된 mean/median/p95와 불일치 {len(mismatches)}건. 음수·비유한 값 및 시간 합산 계약 위반은 0건.",
f"최대 CPU 시간은 {max(r['cpu_max_ms'] for r in audit):.3f}ms이고 CPU 5ms 초과는 {sum(r['cpu_over_5ms'] for r in audit)}스텝이었다. 꼬리값은 제거하지 않았다.",
'cpu_phase_statistics_us.csv에 구간별 mean/median/p95를, timing_audit.csv에 극값과 마감 위반 횟수를 저장했다. 평균 구간 합은 CPU 합계와 일치해야 하지만 구간별 median/p95의 합은 합계의 median/p95와 같을 필요가 없다.',
'step_total은 모델 통신 대기를 포함하므로 CPU 합계와 구분한다. CPU 단일 스레드 설정은 OS 스케줄링 격리를 뜻하지 않는다. 실행 중 별도 테스트·smoke도 수행했으므로 시간적 공존에 따른 간섭 가능성이 있으며 작은 타이밍 차이의 인과 해석에는 한계가 있다.','',
'## 최종 결론 초안','',
'이 Hallway·직렬 링크 모델에서 마감 cutoff는 통신 지연을 메시지 폐기와 캐시 정보 부족으로 전환한다. 혼잡 상한은 slow5ms에서 동일 payload goodput을 유지하면서 송신량과 폐기를 감소시켰다. 제어 성능은 학습 분포와 링크 조건에 의존하며, 통신 인지 학습은 일부 병목에서 평균 Return을 높였으나 여유 있는 조건에서는 동일 학습량 Full-Comm 대조군보다 낮았다. AoI·감쇠·분산 로컬 점수의 일관된 추가 이득은 확인되지 않았다.',
'무통신 조건의 Return 상승만으로 dead-reckoning/open-loop 동기화 학습을 확정할 수 없다. Q-gap이 메시지 수신자의 가치를 직접 측정하지 않으므로 AoI 슬롯 선점 설명 역시 가설로 남는다.',
'새 학습 없이 관측 결과를 문서화했으며 모델 구성의 보편적 우위·실무 무선망 보장·통계적 동등성을 주장하지 않는다.']
(OUT/'FINAL_RESEARCH_REPORT_KO.md').write_text('\n'.join(lines))
sources += [historical,gap/'evaluation.json',gap/'final_tables.json']
(OUT/'manifest.json').write_text(json.dumps(dict(status='completed',training_launched=False,
 sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
 timing_groups=len(audit),steps=sum(r['steps'] for r in audit),timing_mismatches=len(mismatches)),indent=2))
print(OUT)
print('AUDIT',len(audit),sum(r['steps'] for r in audit),max(r['cpu_max_ms'] for r in audit),sum(r['cpu_over_5ms'] for r in audit))
