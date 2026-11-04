from pathlib import Path
import json,html,collections,sys
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'formal'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def esc(x):return html.escape(str(x))
def interval(s,scale=1):return f"{s['point']*scale:.2f} [{s['ci95'][0]*scale:.2f}, {s['ci95'][1]*scale:.2f}]"
def table(h,rows):return '<table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
STYLE='<style>body{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;margin:30px;color:#203040}h1{font-size:28px}h2{margin-top:32px}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border:1px solid #bcc9d4;padding:8px;vertical-align:top}th{background:#eaf1f6}tr:nth-child(even){background:#f6f8fa}a{color:#14619a}p{max-width:1250px}details{margin:18px 0}pre{white-space:pre-wrap}@media print{tr{break-inside:avoid}}</style>'
def doc(p,title,body):p.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+esc(title)+'</title>'+STYLE+'<h1>'+esc(title)+'</h1>'+body+'</html>',encoding='utf-8')
cells=['internvl3_2b__mmstar','internvl3_2b__ai2d','qwen2_5_vl_7b__mmstar','qwen2_5_vl_7b__ai2d'];data=[load(D/(c+'.json')) for c in cells]
assert load(D/'completion.json')['status']=='PASS'
M=[x for d in data for x in d['cross_matrices']];B=[x for d in data for x in d['backend_contrasts']];S=[x for d in data for x in d['risk_source_contrasts']];C=[dict(cell=d['cell'],**x) for d in data for x in d['error_composition']]
assert (len(M),len(B),len(S),len(C))==(432,192,432,8)
def findstat(v):
    if isinstance(v,dict):
        if 'invalid_replicates' in v:yield v
        for x in v.values():yield from findstat(x)
    elif isinstance(v,list):
        for x in v:yield from findstat(x)
stats=list(findstat(data));assert all(x['invalid_replicates']==0 for x in stats)
checks=sum(len(d['verification']['archived_checks_passed']) for d in data);assert checks==132
for name,rows in [('cross_matrices',M),('backend_contrasts',B),('risk_source_contrasts',S),('error_composition',C)]:save(H/(name+'.json'),rows)
def matrix(cell,source,target,signal='max',q=.2):return next(x for x in M if x['cell']==cell and x['risk_source']==source and x['error_target']==target and x['signal']==signal and x['q']==q)
primary=[]
for d in data:
 for policy in ['native risk per backend','BF16 risk']:
  x=next(x for x in B if x['cell']==d['cell'] and x['signal']=='max' and x['q']==.2 and x['policy']==policy)
  t=matrix(d['cell'],'TorchAO' if policy.startswith('native') else 'BF16','TorchAO');h=matrix(d['cell'],'HQQ' if policy.startswith('native') else 'BF16','HQQ')
  primary.append(dict(x,torchao_unreviewed=t['unreviewed_errors']['point'],hqq_unreviewed=h['unreviewed_errors']['point']))
save(H/'primary_backend_comparison.json',primary)
pr=[[x['cell'],x['policy'],int(x['torchao_unreviewed']),int(x['hqq_unreviewed']),interval(x['delta_unreviewed_errors']),interval(x['delta_unreviewed_per1000']),'方向未定' if x['pointwise_interval_direction']=='direction uncertain' else x['pointwise_interval_direction']] for x in primary]
body='<p>日期：2026-11-03。现有缓存上的事后配对图像组分析，CPU执行，四格各10,000次重采样。所有区间为逐点95%百分位区间，未作多重性校正。</p>'
body+='<p><strong>主要结果：</strong>20%预算、最大概率风险下，各自后端风险策略和固定BF16风险策略的八项跨后端未送审错误差区间均包含零，均标记为“方向未定”。本文将后端选择参考呈现为带不确定性的条件化比较。</p>'
body+='<p>指标定义：E为部署后端总错误数；Y为预算内捕获错误数；U=E−Y为未送审基准错误数；每千输入暴露=1000U/n。相同图像组抽样同时作用于全部评分来源和错误目标。后端差方向统一为HQQ−TorchAO，负值对应HQQ未送审错误更少。</p>'
body+='<h2>1．主要后端比较：20%预算、最大概率风险</h2>'+table(['Cell','评分策略','TorchAO未送审','HQQ未送审','差值计数及95%区间','每千输入差及95%区间','区间方向'],pr)
body+='<p>Qwen/AI2D各自风险策略的点差为−12，区间[−29,12]；固定BF16风险策略的点差为+1，区间[−11,11]。这组结果提供策略相关点值变化及其不确定性。</p>'
body+='<h2>2．量化错误构成</h2><p>新增错误：BF16正确而量化端错误；纠正错误：BF16错误而量化端正确；共同错误：两端都错误。恒等关系E_quant=E_BF16+新增−纠正在所有重复中核验。</p>'
body+=table(['Cell','后端','总错误及95%区间','新增错误及95%区间','纠正错误及95%区间','共同错误及95%区间'],[[x['cell'],x['backend'],interval(x['total_errors']),interval(x['new_errors_vs_BF16']),interval(x['corrected_errors_vs_BF16']),interval(x['shared_errors'])] for x in C])
body+='<h2>3．固定错误目标下的风险来源比较</h2><p>评分来源A−B的正捕错差表示A捕获更多该部署目标的错误；使用同次配对抽样计算直接差值。完整表包括三个风险来源配对、三个错误目标、三信号及四预算。</p>'
source_primary=[x for x in S if x['signal']=='max' and x['q']==.2]
body+=table(['Cell','错误目标','A−B','捕错数差及95%区间','每100审阅名额收益差及95%区间'],[[x['cell'],x['error_target'],x['source_a']+'−'+x['source_b'],interval(x['delta_captured_errors']),interval(x['delta_yield_per100'])] for x in source_primary])
body+='<h2>4．预算与评分策略敏感性</h2><p>覆盖5%、10%、20%、30%预算；最大概率、熵、边际三信号；固定BF16、固定TorchAO、固定HQQ、各自后端风险四种策略。方向标签仅反映相应逐点区间；整套敏感性结果完整呈现。</p>'
for cell in cells:
 body+='<details><summary>'+esc(cell)+'：48项跨后端比较</summary>'+table(['信号','预算','策略','未送审错误差及95%区间','每千输入差及95%区间','逐点方向'],[[x['signal'],x['q'],x['policy'],interval(x['delta_unreviewed_errors']),interval(x['delta_unreviewed_per1000']),x['pointwise_interval_direction']] for x in B if x['cell']==cell])+'</details>'
body+='<h2>5．交叉审计画像</h2><p>以下各表固定预算20%、最大概率风险，列出三风险来源×三错误目标的捕错、召回和未送审暴露。完整432行结果另存JSON。</p>'
for cell in cells:
 body+='<details><summary>'+esc(cell)+'：三来源×三目标</summary>'+table(['风险来源','错误目标','总错误','捕错数及95%区间','Error Recall及95%区间','未送审数及95%区间','每千输入暴露及95%区间','随机召回期望','理想召回上界'],[[x['risk_source'],x['error_target'],x['total_errors'],interval(x['captured_errors']),interval(x['error_recall']),interval(x['unreviewed_errors']),interval(x['unreviewed_per1000']),round(x['random_recall'],6),round(x['oracle_recall'],6)] for x in M if x['cell']==cell and x['signal']=='max' and x['q']==.2])+'</details>'
body+='<h2>6．核验与证据</h2><p>48组随机图像重采样以显式展开的重复样本独立核对快速计数；132项历史区间/点值复现检查全部通过。三种恒等式在每次抽样检查。'+str(len(stats))+'项区间估计均无未定义重复。原始输入缓存保持原样。</p>'
body+='<ul>'+''.join('<li><a href="'+name+'">'+name+'</a></li>' for name in ['primary_backend_comparison.json','backend_contrasts.json','risk_source_contrasts.json','error_composition.json','cross_matrices.json','run_cpu_audit.py','formal/analysis_plan.json','completion_receipt.json','manuscript_addition.txt'])+'</ul>'
doc(H/'audit_report.html','固定审阅预算的量化后端风险画像：CPU补充分析',body)
method='We performed a post-hoc paired image-group bootstrap audit using 10,000 replicates per model–dataset cell. Each draw sampled G image groups with replacement, retained all questions and their occurrence multiplicities, and applied the same weights to BF16, TorchAO and HQQ scores and errors. Question-weighted endpoints used the replicate-specific n* and k*=ceil(q n*). We evaluated maximum-probability, entropy and margin risks at 5%, 10%, 20% and 30% capacities. For each score source and deployment error target, we reported total errors E, captured errors Y, Error Recall Y/E and unreviewed error exposure U=E−Y, including 1,000U/n. Direct backend contrasts used HQQ minus TorchAO exposure under fixed score-source policies and native risk per backend. Direct score-source contrasts held the deployment error target fixed. All 95% percentile intervals were pointwise and unadjusted for multiplicity. Ties followed descending risk then ascending canonical source row.'
result='At 20% capacity under native maximum-probability risk, the HQQ-minus-TorchAO unreviewed-error differences were +17 (95% image-group interval [−16,52]), +26 [−11,62], +9 [−14,25] and −12 [−29,12] in the order InternVL/MMStar, InternVL/AI2D, Qwen/MMStar and Qwen/AI2D. Under fixed BF16 risk, the corresponding differences were −7 [−38,26], −10 [−43,27], −4 [−21,12] and +1 [−11,11]. All eight primary intervals included zero. The audit reports strategy-dependent point estimates together with unresolved comparison directions at this capacity.'
composition='Error composition was partitioned relative to BF16 into newly erroneous, corrected, shared-error and shared-correct inputs. For InternVL/AI2D, TorchAO introduced 275 errors and corrected 155 BF16 errors; HQQ introduced 285 and corrected 126. The totals were 841 and 880. This accounts for total-error changes alongside the separate budgeted capture and residual exposure endpoints.'
text='Methods addition\n'+method+'\n\nResults addition\n'+result+'\n\nError composition addition\n'+composition+'\n\n中文解释：后端比较以相同输入、预算和明确的评分策略为条件；区间方向未定也是审计输出。三类指标通过U=E−Y衔接，新增和纠正错误通过样本级配对统计。\n'
(H/'manuscript_addition.txt').write_text(text,encoding='utf-8')
(H/'response_addition.txt').write_text('The supplementary CPU audit now directly compares backend unreviewed exposure, crosses all three risk sources with all three error targets, reports error creation/correction composition and evaluates all four review budgets and three risk signals. '+result+' '+method,encoding='utf-8')
save(H/'completion_receipt.json',{'status':'COMPLETE','date':'2026-11-03','gpu_inference':False,'cells':4,'replicates_per_cell':10000,'matrix_rows':len(M),'backend_contrast_rows':len(B),'risk_source_contrast_rows':len(S),'composition_rows':len(C),'interval_estimands':len(stats),'invalid_replicates':0,'archived_reproduction_checks_passed':checks,'independent_expanded_occurrence_cases':48,'primary_eight_intervals_include_zero':all(x['delta_unreviewed_errors']['ci95'][0]<=0<=x['delta_unreviewed_errors']['ci95'][1] for x in primary),'design':'post-hoc; pointwise 95%; unadjusted multiplicity','report':'audit_report.html','environment':{'python':sys.version,'numpy':'recorded in formal/analysis_plan.json'}})
# Attach concrete artifacts to the relevant analysis directories via stable relative links.
mapping={'Analysis_1/Major_1':'端点解释与后端直接比较','Analysis_1/Major_2':'四预算区间','Analysis_1/Major_3':'三信号敏感性','Analysis_1/Major_5':'召回参照与派生结果','Analysis_2/Major_1':'交叉审计贡献与错误组成','Analysis_2/Major_2':'HQQ跨后端比较','Analysis_2/Major_3':'统一图像组区间','Analysis_2/Major_4':'分析代码与复现记录'}
for folder,role in mapping.items():
 d=ROOT/'revise'/folder
 doc(d/'cpu_audit_addendum_20261103.html',role+'：CPU补充分析已完成','<p>2026-11-03。'+esc(role)+'对应结果已计算完成。四格各10000次配对图像组抽样，132项历史复现检查通过。</p><p><a href="../../CPU_backend_audit_20261103/audit_report.html">完整报告与数据</a> · <a href="../../CPU_backend_audit_20261103/manuscript_addition.txt">方法与结果替换文字</a> · <a href="../../CPU_backend_audit_20261103/response_addition.txt">回复增补草稿</a></p><p>主结果：20%最大概率风险下，八项主要跨后端比较均为方向未定。主稿文字与最终表号在整合阶段统一。</p>')
 save(d/'cpu_audit_addendum_status_20261103.json',{'status':'COMPLETE','role':role,'report':'../../CPU_backend_audit_20261103/audit_report.html','replicates_per_cell':10000,'main_docx_merge':'pending'})
 p=d/'implementation.html'
 if p.exists():
  s=p.read_text(encoding='utf-8');addition='<p><a href="cpu_audit_addendum_20261103.html">2026-11-03：后端画像CPU补充分析已完成</a></p>'
  if addition not in s:p.write_text(s+addition,encoding='utf-8')
index=ROOT/'revise/index.html';s=index.read_text(encoding='utf-8');link='<p><a href="CPU_backend_audit_20261103/audit_report.html">2026-11-03：后端风险画像与直接差值CPU分析，全部完成</a></p>'
if link not in s:index.write_text(s+link,encoding='utf-8')
gpath=ROOT/'revise/implementation_status.json';g=load(gpath);g['cpu_backend_audit_20261103']={'status':'COMPLETE','report':'CPU_backend_audit_20261103/audit_report.html','archived_checks':132,'primary_direction':'all eight comparisons uncertain'};save(gpath,g)
print(json.dumps(load(H/'completion_receipt.json'),ensure_ascii=False))
