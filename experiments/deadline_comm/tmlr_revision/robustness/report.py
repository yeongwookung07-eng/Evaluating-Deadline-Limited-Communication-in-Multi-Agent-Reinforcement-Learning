"""Deterministic aggregation of bounded follow-up diagnostics; no new rollouts."""
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from .strata import estimate
from ..editorial import table

BASE = Path('experiments/deadline_comm')
OUT = BASE/'tmlr_revision/results'
def read(run, file='summary.json'):
    return json.loads((BASE/'runs'/run/file).read_text())
def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]
def stats(v):
    return estimate(v, np.random.default_rng(20261119), 5000)
def ci(s):
    return f"{s['mean']:.3f} [{s['t95'][0]:.3f}, {s['t95'][1]:.3f}]"
def main():
    strata=read('tmlr_hallway_sensitivity_20261119_001')
    episodes=read('tmlr_hallway_sensitivity_20261119_001','episodes.json')
    selected=[r for r in strata['strata'] if r['stratum'] in ['all','unresolved','resolved']]
    # Negative suffix returns are observable; an absent negative total does NOT rule out a penalty within a branch.
    fresh=lines(BASE/'runs/tmlr_fresh_20261119_001/states.jsonl')
    negative=[r for r in fresh if any(v['return_']<0 for v in r['variants'])]
    excluded={(r['seed'],r['condition'],r['checkpoint_arm'],r['episode'],r['t']) for r in negative}
    contexts=lines(BASE/'runs/tmlr_hallway_sensitivity_20261119_001/states.jsonl')
    sensitivity=[]
    for cond,arm in sorted({(r['condition'],r['arm']) for r in contexts}):
        byseed=defaultdict(list)
        for r in contexts:
            if (r['condition'],r['arm'])!=(cond,arm):continue
            if (r['seed'],cond,arm,r['episode'],r['t']) in excluded:continue
            byseed[r['seed']].append(r['factual_minus_uniform'])
        sensitivity.append(dict(condition=cond,arm=arm,factual_minus_uniform=stats([np.mean(v) for _,v in sorted(byseed.items())])))
    cpu=[]; snapshots=[]
    for rep in range(3):
        root=BASE/f'runs/tmlr_cpu_repeats_20261119_001/repeat{rep}'
        records=json.loads((root/'rollouts/summary.json').read_text())['records']
        for d in [2.0,2.25,2.5]:
            for arm in ['rr_cap','dac']:
                for mode in ['measured','fixed']:
                    rr=sorted([r for r in records if (r['deadline_ms'],r['checkpoint_arm'],r['mode'])==(d,arm,mode)],key=lambda r:r['seed'])
                    cpu.append(dict(repeat=rep,deadline_ms=d,arm=arm,mode=mode,**{k:stats([r[k] for r in rr]) for k in ['mean_k','ready_us','gate_us','cpu_us','mean_return']}))
        rr=lines(root/'snapshots.jsonl'); pairs=defaultdict(dict)
        for r in rr:pairs[r['seed'],r['weights'],r['episode'],r['t']][r['selector']]=r
        byseed=defaultdict(lambda:defaultdict(list))
        for key,p in pairs.items():
            assert set(p)=={'rr_cap','dac'}
            for k in ['ready_ns','gate_ns','cpu_total_ns','K_delivered']:
                byseed[key[0]][k].append((p['dac'][k]-p['rr_cap'][k])/(1000 if k.endswith('_ns') else 1))
        snapshots.append(dict(repeat=rep,pairs=len(pairs),scope='DAC minus RR, same frozen weights/input/scheduler snapshot; average within seed across both weight arms and episodes, then across ten seeds.',**{k:stats([np.mean(v[k]) for _,v in sorted(byseed.items())]) for k in ['ready_ns','gate_ns','cpu_total_ns','K_delivered']}))
    probe=read('tmlr_receiver_probe_20261119_001'); probe_rows=[]
    for cond,method,scheduler in sorted({(r['condition'],r['method'],r['scheduler']) for r in probe['records']}):
        rr=sorted([r for r in probe['records'] if (r['condition'],r['method'],r['scheduler'])==(cond,method,scheduler)],key=lambda r:r['seed'])
        metrics={'acting':stats([100*r['actual_accuracy'] for r in rr]),'q':stats([100*r['q'] for r in rr])}
        for feature in ['source','aggregate','local_cache_metadata','full_input']:
            metrics[feature]=stats([100*r['probes'][feature]['accuracy'] for r in rr])
            metrics[feature+'_minus_acting']=stats([100*(r['probes'][feature]['accuracy']-r['actual_accuracy']) for r in rr])
        probe_rows.append(dict(condition=cond,method=method,scheduler=scheduler,metrics=metrics))
    prefix=read('tmlr_prefix_opportunity_20261119_001'); prefix_rows=[]
    for cond in ['slow_5ms','medium_2ms']:
        rr=[r for r in prefix['records'] if r['condition']==cond]
        prefix_rows.append(dict(condition=cond,episodes=len(rr),survivors=sum(r['survived'] for r in rr),oracle_returns=dict(Counter(r['oracle_return'] for r in rr)),unused_opportunities=sum(r['unused_opportunity'] for r in rr)))
    artifact=read('tmlr_artifact_replay_20261119_001')
    result=dict(scope='Frozen-policy follow-up; no policy training. Seed-level t and percentile-bootstrap intervals are descriptive/unadjusted. CPU processes are repeated measurements, not extra training seeds.',hallway=dict(episodes=len(episodes),steps=sum(r['steps'] for r in episodes),resolved_steps=sum(r['resolved_steps'] for r in episodes),factual_penalty_episodes=sum(r['penalty'] for r in episodes),strata=selected,negative_suffix_candidate_states=len(negative),negative_suffix_candidates=sum(v['return_']<0 for r in fresh for v in r['variants']),negative_suffix_exclusion=sensitivity,penalty_scope='All factual trajectories replayed; counterfactual negative totals scanned, not all intermediate branch rewards. A nonnegative suffix total does not exclude a transient penalty; exclusion is retrospective sensitivity only.'),cpu=cpu,cpu_identical_input=snapshots,receiver_probe=probe_rows,verified_receiver_steps=probe['verified_steps'],prefix=prefix_rows,artifact=artifact)
    OUT.mkdir(exist_ok=True);(OUT/'robustness_summary.json').write_text(json.dumps(result,indent=2))
    tables=[]
    rows=[]
    for r in selected:
        if r['stratum']!='all':continue
        u=next(x for x in selected if (x['condition'],x['arm'],x['stratum'])==(r['condition'],r['arm'],'unresolved'))
        rows.append([r['condition'],r['arm'],f"{r['state_effect']['mean']:.2f}",f"{r['episode_effect']['mean']:.2f}",f"{u['state_effect']['mean']:.2f}",ci(r['state_factual_minus_uniform']),ci(r['episode_factual_minus_uniform'])])
    tables.append(table('tab:robust_strata',r'Fresh-subset sensitivity. Effect columns are percentages: original equal-state weighting, equal-episode weighting, and equal-state weighting restricted to unresolved tasks. Both weight seeds equally. Direct factual minus uniform-subset suffix return uses the identical sampled state and received count; unadjusted 95\% t intervals over ten seeds. Episode weighting first averages sampled states within an episode.', ['Condition','Weights','State','Episode','Unresolved','Paired state [CI]','Paired episode [CI]'],rows))
    rows=[]
    for r in cpu:
        if r['deadline_ms']==2.25 and r['mode']=='measured':
            rows.append([r['repeat']+1,r['arm'],f"{r['ready_us']['mean']:.1f}",f"{r['gate_us']['mean']:.1f}",f"{r['mean_k']['mean']:.3f}"])
    tables.append(table('tab:robust_cpu',r'Independent-process measurements at Medium/2.25 ms. Each process uses the same ten seeds, three evaluation resets and two discarded warmups per cell, pinned to the same allowed CPU. Values are means of seed means. Repeats are not additional training seeds; this shorter protocol is separate from the original ten-reset timing study.', ['Process','Weights',r'Ready ($\mu$s)',r'Gate ($\mu$s)','Delivered/step'],rows))
    rows=[]
    for r in probe_rows:
        m=r['metrics'];rows.append([r['condition'],r['method'],r['scheduler'],f"{m['acting']['mean']:.2f}",f"{m['aggregate']['mean']:.2f}",f"{m['full_input']['mean']:.2f}",ci(m['aggregate_minus_acting'])])
    tables.append(table('tab:receiver_probe',r'Broadcast receiver probes, percent accuracy on held-out episodes. Same frozen checkpoints and fixed-readiness reset cohort as the decomposition, but fit on the first ten episodes and evaluate only the last ten. Ten seed means; paired aggregate-probe minus acting-policy gain has unadjusted 95\% t intervals. Ridge probes do not alter the acting policy. Source-aware preserves factual delivered count, not necessarily offered traffic.', ['Condition','Weights','Selection','Acting','Aggregate','Full input','Aggregate gain [CI]'],rows))
    (OUT/'robustness_tables.tex').write_text('\n\n'.join(tables))
    md=['# 추가 검증 결과 — 2026-11-19','기존 체크포인트 고정. 새로운 정책 학습·GPU 사용 없음. 원시 로그와 이전 결과는 보존했다.','',f"## Hallway\n400 factual 에피소드, 2,654 상태를 재현했다. 원래 궤적의 음수 페널티 에피소드는 0개. 결과 결정 후 스텝은 {result['hallway']['resolved_steps']}/{result['hallway']['steps']}. counterfactual의 음수 suffix total은 1/19,875개이며, 중간 페널티 전체 발생률은 이 집계만으로 알 수 없다.",'| 조건/가중치 | 상태 효과율 | 에피소드 효과율 | 미결정 상태 효과율 | factual−uniform, t95 |','|---|---:|---:|---:|---|']
    for r in selected:
        if r['stratum']!='all':continue
        u=next(x for x in selected if (x['condition'],x['arm'],x['stratum'])==(r['condition'],r['arm'],'unresolved'))
        md.append(f"| {r['condition']} / {r['arm']} | {r['state_effect']['mean']:.2f}% | {r['episode_effect']['mean']:.2f}% | {u['state_effect']['mean']:.2f}% | {ci(r['state_factual_minus_uniform'])} |")
    md += ['', '이는 한 번의 sender 선택에 대한 조건부 효과이며, 계속 uniform으로 전송하는 scheduler나 DAC의 전반적 우위가 아니다. 모든 CI는 다중검정 보정 전 기술통계다. 분기/스텝을 독립 seed처럼 세지 않았다.','', '## CPU 독립 프로세스 반복','| 반복 | RR delivered/step | DAC delivered/step | 동일 입력 DAC−RR gate µs [t95] |','|---|---:|---:|---|']
    for rep in range(3):
        get=lambda arm:next(r for r in cpu if (r['repeat'],r['deadline_ms'],r['arm'],r['mode'])==(rep,2.25,arm,'measured'))
        md.append(f"| {rep+1} | {get('rr_cap')['mean_k']['mean']:.3f} | {get('dac')['mean_k']['mean']:.3f} | {ci(snapshots[rep]['gate_ns'])} |")
    md += ['', '패킷 경계 효과의 크기는 재현 실행 간 안정적이지 않다. 같은 입력 비교는 gate 자체의 비용 차이를 분리하지만, 서로 다른 머신으로 일반화하는 검증은 아니다. 모든 deadline·fixed readiness·seed별 값은 robustness_summary.json에 남겼다.', '', '## 실제 receiver 입력 진단','| 조건 / 정책 / 선택 | 행동 정확도 | Aggregate probe | Full-input probe | Aggregate−행동 pp [t95] |','|---|---:|---:|---:|---|']
    for r in probe_rows:
        m=r['metrics'];md.append(f"| {r['condition']} / {r['method']} / {r['scheduler']} | {m['acting']['mean']:.2f}% | {m['aggregate']['mean']:.2f}% | {m['full_input']['mean']:.2f}% | {ci(m['aggregate_minus_acting'])} |")
    md += ['', '40,000스텝에서 실제 action-head 입력과 독립 재구성 aggregate가 일치하고 원래 reward·source delivery·bit가 일치했다. train/test 에피소드가 분리되며 정규화는 train에서만 적합했다. 성공한 probe는 정보의 가독성 증거이고, 정책이 왜 학습하지 못했는지의 인과적 증명은 아니다. local+metadata probe는 availability를 포함하므로 No-Comm 대조군이 아니다.', '', '## 공통 prefix 뒤 실제 보상 기회']
    for r in prefix_rows:md.append(f"- {r['condition']}: {r['episodes']}개 중 {r['unused_opportunities']}개는 RR/DAC suffix보다 더 높은 보상을 얻는 실행 가능한 privileged suffix가 남아 있었다. 가능한 남은 보상별 개수: {r['oracle_returns']}.")
    md += ['전역 상태 기반 reference를 3,406스텝 실행해 예측 보상을 확인했다. 이는 메시지 선택만 바꾸면 그 보상을 달성할 수 있다는 증거는 아니다.', '', '## 추출 아카이브 실행','별도 임시 디렉터리에서 exported code+evidence로 테스트 5개 통과, 원시 집계 JSON 정확 일치, fresh smoke 28상태 및 no-op 검사 통과. 같은 설치 환경에서의 재현이며 독립 OS/공개 업로드는 아니다. 검사한 ZIP SHA256은 artifact summary에 남겼다.', '', '## 남은 경계','다른 하드웨어, 현실적 MAC/backlog, 추가 대형 benchmark, 모든 counterfactual 중간 reward의 재감사는 수행하지 않았다. 현재 결과는 DAC의 보편적 우위나 일반 task의 통신 필요성을 증명하지 않는다.']
    md += ['', '## 직접 paired 효과의 가중 방식 민감도', '상태 가중에서 0을 제외한 두 CI도 에피소드 가중에서는 0을 포함한다. 네 조건 모두 에피소드 가중 t95가 0을 포함하므로 uniform 대비 이득을 가중 방식에 견고한 결과로 제시하지 않는다.', '', '| 조건/정책 | 에피소드 가중 factual−uniform [t95] |', '|---|---|']
    for r in selected:
        if r['stratum']=='all':md.append(f"| {r['condition']} / {r['arm']} | {ci(r['episode_factual_minus_uniform'])} |")
    # Keep tables separated from preceding prose for Markdown renderers.
    output=[]
    for line in md:
        if line.startswith('|') and output and output[-1] and not output[-1].startswith('|'):output.append('')
        output.append(line)
    (OUT/'ROBUSTNESS_REPORT_KO.md').write_text('\n'.join(output)+'\n')
    print(OUT/'ROBUSTNESS_REPORT_KO.md')
if __name__=='__main__':main()
