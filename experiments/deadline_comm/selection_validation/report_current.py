"""Consolidate only completed evidence; explicitly mark pending primary contrasts."""
import json
from pathlib import Path
from ..reviewer_validation.stats import paired_bootstrap

root=Path('experiments/deadline_comm/runs')
source=root/'training_controls_20261118_001_evaluation_verified_002/summary.json'
d=json.loads(source.read_text());agg=d['aggregates'];records=d['records']
conditions=['slow_5ms','medium_2ms','medium_5ms','medium_1.5ms']
lines=['# 当前结果核验 — 2026-11-18','',f'已完成来源：{source}。24个学训练任务、132个最终评估组合。以下均保留3个学习seed，mean ± sample SD；不是10-seed结果。','',
'## 四项主检验状态','',
'1. Endpoint：本评估正在运行；No-Comm/Ideal/DAC完整3-seed比较尚未产出，不引用smoke分数。',
'2. 24个训练任务：已全部完成，最终评估132/132完成。见下表。',
'3. 10-seed：尚未完成；不得用下面的3-seed区间替代。',
'4. 相同cap Round-Robin：实现/测试/smoke完成，排在10-seed学训练前；RR亦加入从头训练。历史48B Periodic比较另列，不混作新实验。','',
'## 同学训练量的继续训练（全部以DAC部署）','',
'|训练方式|slow5|medium2|medium5|medium1.5|','|---|---|---|---|---|']
def values(family,arm,deployment):
 rows=[]
 for c in conditions:
  r=next(r for r in agg if (r['family'],r['arm'],r['deployment'],r['condition'])==(family,arm,deployment,c))
  rows.append(f"{r['mean_return']:.2f} ± {r['mean_return_sd']:.2f}")
 return '|'.join(rows)
for arm,label in [('full_continuation','Full继续训练 + DAC部署'),('dac','DAC通信感知继续训练')]:lines.append('|'+label+'|'+values('continuation',arm,'dac')+'|')
lines+=['','## 适应阶段训练Ablation：训练和部署均去除同一模块','', '|设置|slow5|medium2|medium5|medium1.5|','|---|---|---|---|---|']
for arm,label in [('dac','完整DAC'),('dac_no_cap','无link cap，保留预算cap'),('dac_no_aoi','无发送AoI项'),('dac_no_decay','无幅度感衰，保留age/valid特征')]:lines.append('|'+label+'|'+values('continuation',arm,arm)+'|')
lines+=['','这些结果不支持所有模块均必需。AoI移除后的均值在四条件均较高，但3个seed不证明普遍优势。适应阶段移除不等于从初始化开始移除。完整CSV另含将各消融checkpoint重新以完整DAC部署的比较，用于分离模型和部署规则。','',
'## 从头学训练：所有方法均在slow5训练，1500updates、同Oracle/BC协议','', '|方式|slow5|medium2|medium5|medium1.5|','|---|---|---|---|---|']
for arm in ['full','periodic','dac']:lines.append('|'+arm+'|'+values('scratch',arm,arm)+'|')
lines+=['','slow5中三种方法均值接近；medium2的DAC均值较高。不能据此推断10-seed或无Oracle结论。','', '## 历史48B对照的逐seed证据','']
old=json.loads((root/'day3_fair_budget_20261118_002/records.json').read_text())
a={r['seed']:r['mean_return'] for r in old if r['condition']=='slow_5ms' and r['budget']==48 and r['method']=='dac'}
b={r['seed']:r['mean_return'] for r in old if r['condition']=='slow_5ms' and r['budget']==48 and r['method']=='periodic'}
old_ci=paired_bootstrap(a,b)
lines+=['DAC Return: '+str(a),'','Periodic Return: '+str(b),'',f"DAC-Periodic均值差 {old_ci['mean_difference']:.2f}，3-seed描述性paired bootstrap95%区间 [{old_ci['ci95_low']:.2f}, {old_ci['ci95_high']:.2f}]。差异主要来自seed0；不能确认稳定优势。双方均48B offered、32B payload、0%废弃。",'',
'## 其他已完成与未完成','',
'- 900条件旧评估及Figure1–3、英文稿/20篇引用完成；原结论为容量约束下效率与控制表现的权衡。',
'- 最新网络720组diagnostic和hidden task80组sanity完成，但不是学到的N50控制结果。',
'- Native SchedNet和RR700-step smoke成功，不等于50k-step本学训练或同deadline原算法比较。',
'- 真实网络trace、同deadline的native对DAC仍未完成。',
'- 新的10-seed、额外环境和规模结果不能预先写入论文。']
ci=[]
for family,a,b,da,db in [('continuation','dac','full_continuation','dac','dac'),('scratch','dac','periodic','dac','periodic')]:
 for c in conditions:
  left={r['seed']:r['mean_return'] for r in records if (r['family'],r['arm'],r['deployment'],r['condition'])==(family,a,da,c)}
  right={r['seed']:r['mean_return'] for r in records if (r['family'],r['arm'],r['deployment'],r['condition'])==(family,b,db,c)}
  ci.append(dict(family=family,condition=c,contrast=a+' minus '+b,**paired_bootstrap(left,right)))
out=Path('experiments/deadline_comm/selection_validation')
(out/'CURRENT_RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
(out/'current_paired_ci.json').write_text(json.dumps(dict(current_3seed=ci,historical_48B=old_ci),indent=2)+'\n')
print(json.dumps(ci,indent=2))
