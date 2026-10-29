"""Render final evidence, including negative results and synthetic limitations."""
import argparse,csv,json,os,sys
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from citra.final_diagnosis import protocol_config_path


def read(path):return json.loads(Path(path).read_text())
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('|'+ '|'.join(map(str,row))+'|' for row in rows)+'\n'
def ms(x):return f'{x*1000:.3f}'
def meanstd(d):return f"{d['mean']:.6f} ± {d['std']:.6f}"


def figures(root,counts,system,semantic,scales):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'figures';out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((root/'system/document_counts.csv').open()))
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for year in sorted(counts['years']):
        vals=[int(r['legal_window_documents']) for r in rows if r['year']==year]
        ax.hist(vals,bins=np.arange(11.5,48.5),histtype='step',label=year,density=True)
    ax.set(xlabel='Legal visible documents in (cutoff - 7 days, cutoff]',ylabel='Frequency density',title='Actual forecast origins, not raw archive file counts');ax.legend()
    fig.savefig(out/'real_window_counts.png',dpi=160);fig.savefig(out/'real_window_counts.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    labels=[]
    for i,r in enumerate(system['summary']):
        labels.append(r['method']+(f" eps={r['epsilon']:g}" if r['epsilon'] is not None else ''))
        for key,color,mark in (('p50','C0','o'),('p95','C1','s'),('p99','C2','^')):
            ax.scatter(i,r['timings']['end_to_end_seconds'][key]*1000,c=color,marker=mark,label=key if i==0 else None)
    ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='End-to-end latency (ms)',title='As-of prefix states; same frozen checkpoint/cache and CPU query/forecast')
    ax.tick_params(axis='x',rotation=20);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'real_asof_latency.png',dpi=160);fig.savefig(out/'real_asof_latency.pdf');plt.close(fig)
    names=['ts_only','real_text','metadata_only','content_only','constant_text','null_no_metadata','shuffle_retrained','shuffle_fixed_weights']
    years=sorted({g['year'] for g in semantic['groups']});fig,axes=plt.subplots(1,len(years),figsize=(16,5),constrained_layout=True)
    for ax,year in zip(np.atleast_1d(axes),years):
        for i,name in enumerate(names):
            g=next(g for g in semantic['groups'] if g['year']==year and g['kind']==name)['metrics']['normalized_mse']
            ax.errorbar(i,g['mean'],yerr=g['std'],fmt='o',capsize=3)
        ax.set(title=f'{year} '+('new after lock' if year==2024 else 'previously seen'),xticks=range(len(names)),xticklabels=names,ylabel='Normalized MSE; training-seed std')
        ax.tick_params(axis='x',rotation=80);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'semantic_controls.png',dpi=160);fig.savefig(out/'semantic_controls.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,len(years),figsize=(15,6),constrained_layout=True)
    contrasts=[n for n in names if n!='real_text']
    for ax,year in zip(np.atleast_1d(axes),years):
        for i,name in enumerate(contrasts):
            for block,offset,color in ((7,-.12,'C0'),(14,.12,'C1')):
                r=next(r for r in semantic['paired'] if r['year']==year and r['contrast']=='real_text minus '+name and r['metric']=='normalized_mse' and r['block_days']==block)
                ax.plot([r['ci95_low'],r['ci95_high']],[i+offset]*2,c=color);ax.scatter(r['mean_difference'],i+offset,c=color,s=16,label=f'{block}-day blocks' if i==0 else None)
        ax.axvline(0,c='black',lw=1);ax.set(title=str(year),yticks=range(len(contrasts)),yticklabels=contrasts,xlabel='Real minus control MSE (negative favors real)');ax.legend(fontsize=8)
    fig.savefig(out/'paired_block_intervals.png',dpi=160);fig.savefig(out/'paired_block_intervals.pdf');plt.close(fig)
    if scales:
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,scenario in zip(axes,('active_only','mixed_archive')):
            selected=sorted([r for r in scales if r['scenario']==scenario],key=lambda r:r['legal_n'])
            for method,e in [('numpy_exact',None),('citra_avl',0.),('citra_avl',.01),('citra_avl',.1),('citra_avl',1.)]:
                ax.loglog([r['legal_n'] for r in selected],[next(x for x in r['summary'] if x['method']==method and x['epsilon']==e)['total']['p50']*1000 for r in selected],marker='o',label=method+(f' {e:g}' if e is not None else ''))
            ax.set(title='SYNTHETIC: '+scenario,xlabel='Legal N',ylabel='Aggregation pipeline P50 (ms)');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.savefig(out/'synthetic_scaling_latency.png',dpi=160);fig.savefig(out/'synthetic_scaling_latency.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for scenario in ('active_only','mixed_archive'):
            rs=sorted([r for r in scales if r['scenario']==scenario],key=lambda r:r['legal_n'])
            axes[0].loglog([r['legal_n'] for r in rs],[r['construction_seconds'] for r in rs],marker='o',label=scenario)
            axes[1].loglog([r['legal_n'] for r in rs],[r['memory']['index_resident_bytes']/1e6 for r in rs],marker='o',label=scenario+' index')
            axes[1].loglog([r['legal_n'] for r in rs],[r['memory']['process_peak_rss']/1e6 for r in rs],marker='x',linestyle='--',label=scenario+' process peak')
        axes[0].set(xlabel='Synthetic legal N',ylabel='Existing AVL construction (s)');axes[1].set(xlabel='Synthetic legal N',ylabel='Index resident estimate (MB)')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.savefig(out/'synthetic_build_memory.png',dpi=160);fig.savefig(out/'synthetic_build_memory.pdf');plt.close(fig)


def reports(cfg,root,docs,counts,system,full,semantic,scales):
    link=lambda file:os.path.relpath(root/file,docs)
    reg=read(root/'registration.json');raw=read(root/'system_asof/query_records.json');maint=read(root/'system_asof/maintenance.json')
    eps01=next(r for r in system['summary'] if r['method']=='citra_avl' and r['epsilon']==.1)
    npref=next(r for r in system['summary'] if r['method']=='numpy_exact')
    slow=eps01['timings']['end_to_end_seconds']['p50']/npref['timings']['end_to_end_seconds']['p50']
    system_text=f'''# CITRA 最终系统可行性诊断（2026-10-29）

**System No-Go。** 当前真实七天窗口仅约42篇文档；同算子、同冻结参数、同缓存的强向量化Exact更快。
主表使用**只含cutoff当时已公开记录的前缀库**，不以未来归档造成的额外混合节点惩罚AVL。
另保留完整2020–2023归档回放结果作敏感性分析，不能将该回放的未来树结构成本称作真实在线部署成本。

## 协议与实际集合规模

基线commit `{reg['git_commit']}`，分支 `{reg['git_branch']}`。
固定seed7的Window7 Count Mean Residual checkpoint，2020训练、2021验证、2022/2023均为已见诊断；
本系统测量没有使用2024预测结果选查询。每年6个均匀排名起点，共24个真实查询，每法每点10次热调用；
方法顺序按查询/重复循环轮换，不挑选最有利延迟。
2020查询是已拟合模型的回顾性计算，不声称2020在线部署过该模型。

原始发行/观测首次到达尚未获证，仍为issuance_asof_assumption：文本发行+0h、数值观测+1h。
每个cutoff先做权威版本选择，再取公开可用时间在(t−7天,t]的全部合法文档，完整核和除以max(N,1)。
逐查询数量经原始snapshot独立核对，不以七天ZIP产品文件数代替合法N。
'''
    system_text+=table(['年份','实际起点','均值 / 标准差','P50 / P95 / P99 / Max','空窗口比例'],[
        [year,str(v['legal_documents']['n']),f"{v['legal_documents']['mean']:.3f} / {v['legal_documents']['std']:.3f}",
         ' / '.join(f"{v['legal_documents'][k]:g}" for k in ('p50','p95','p99','max')),f"{v['empty_ratio']:.2%}"] for year,v in counts['years'].items()])
    system_text+=f'''
总计{counts['actual_origins']}个日频预测起点，N均值{counts['all']['mean']:.3f}、P95={counts['all']['p95']:g}、最大{counts['all']['max']:g}。
源语料8766个保留文档版本；不确定修订/冲突在准备阶段隔离，不能据“已保留数据的版本过滤前后相同”宣称档案不存在修订。
2020初始查询没有2019年归档，最小N=12来自语料左边界，不能解释为覆盖完整七天的典型产品量；该边界查询保留，没有为了负结论删除。
CSV分别记录可访问候选、版本选择后的全历史数量、窗口前后数量、元数据候选行和Exact Key读取数。
Exact只对合法N个Key/Impact计算核；筛选缓存仍会检查当前前缀的元数据数组，二者不是同一“扫描数”。
[逐查询数量CSV]({link('system/document_counts.csv')})、[数量JSON]({link('system/document_counts.json')})。

![真实合法窗口规模]({link('figures/real_window_counts.png')})

## 同条件强Exact与AVL

共享实际冻结float64 Key/Impact字节和时间/版本元数据缓存；Exact与AVL均不在线重做MiniLM或Key/Impact编码。
所有方法的Query Encoder、数值骨干和最终修正在CPU；CUDA方法**只卸载聚合**，输入query/索引和输出聚合传输均计时并同步。
这使各方法使用逐元素相同的查询、表示和输出形状[1,6,2]，不是不同GPU神经模型的比较，也不是新的Hybrid Index。
独立exact_oracle在正常计时之外核对全部方法、每次运行的输出；不把Top-K或全历史Sum当成当前Exact。
'''
    system_text+=table(['方法 / 输出ε','端到端P50(ms)','P95','P99','Max','聚合或树查询P50(ms)'],[
        [r['method']+(f" / {r['epsilon']:g}" if r['epsilon'] is not None else ''),
         *[ms(r['timings']['end_to_end_seconds'][k]) for k in ('p50','p95','p99','max')],ms(r['timings']['aggregation_seconds']['p50'])] for r in system['summary']])
    system_text+=f'''
ε=.1的AVL端到端P50为NumPy Exact的约{slow:.2f}倍。ε=0并不加速。
背景中的旧1.223ms来自全历史Full的端到端表，不能与本轮纯聚合0.035ms直接相除；本轮结论只取同算子同环境的表内比较。
冷调用单独记录（初始化之后的首次调用，不是全新OS进程/冷磁盘）；每个查询有2次不进入热分位数的预热，预热成本单列。
独立Oracle验证已先触达Query/Forecast和部分算子，所以cold只指首个被测方法调用，不是完全未预热的网络/BLAS；没有声称测得独立进程冷启动尾延迟。
输入历史处理、模型字节校验、时间过滤、Query Encoding、gather/转换、H2D/D2H、聚合、预测修正与最终输出均逐次保存。
数据访问、共享模型保护开销没有只加给某一个方法；CSV可重算P50/P95/P99/Max。
NumPy聚合列不含另列的gather/时间过滤，AVL树查询列包含遍历/检查/聚合，完整成本比较应看端到端列；不把它们当作相同纯算术核。
没有并行GPU训练，但数据收集短暂重叠第二遍测量，且不是独占机器；采用轮换顺序和原始极值保留，而非声称严格硬件隔离。
[主表完整分项JSON]({link('system_asof/summary.json')})、[逐次CSV]({link('system_asof/query_records.csv')})。

![同语义真实前缀库耗时]({link('figures/real_asof_latency.png')})

## 扫描、时间混合和认证预算

ε约束标准化单样本全部horizon/target的**输出L2差**，不是MSE。
归一分母与独立Oracle相同；由实际||W||F换算原始核和预算ε·N/||W||F，归一聚合界再传到预测输出。
这些是float64容差验证的解析余项保证，非形式化IEEE总舍入误差证明。
'''
    diag_rows=[]
    for e in cfg['system']['epsilons']:
        rs=[r for r in raw if r['method']=='citra_avl' and r['epsilon']==e and r['phase']=='warm']
        diag_rows.append([f'{e:g}',f"{np.mean([r['scan_fraction'] for r in rs]):.2%}",
            f"{sum(r['documents_scanned'] for r in rs)/sum(r['legal_documents'] for r in rs):.2%}",
            f"{np.mean([r['query_stats']['nodes_visited'] for r in rs]):.1f}",
            f"{np.mean([r['query_stats']['temporal_visible_nodes'] for r in rs]):.1f}",
            f"{np.mean([r['query_stats']['temporal_partial_nodes'] for r in rs]):.1f}",
            str(sum(r['query_stats']['refinements'] for r in rs)),
            str(sum(r['query_stats']['fallback_exact'] for r in rs)),
            f"{max(r['actual_output_deviation'] for r in rs):.6g} / {max(r['certified_output_bound'] for r in rs):.6g}"])
    system_text+=table(['ε','平均查询扫描比','按文档加权扫描比','访问节点均值','完全合法节点均值','时间混合节点均值','误差细化总数','全精确退化次数/240','最大误差/最大界'],diag_rows)
    system_text+='''
时间混合节点不能直接使用摘要，必须下钻；混合叶的元数据检查和逐点计算单独有计数。
宽预算下仍几乎全扫描，误差界细化为0，故本工作负载不能把代价主要归因于“界太紧”。
树遍历/叶级fsum与函数调用的组合远高于几十篇文档的批量矩阵计算，与Python执行开销一致；
但没有原生实现受控消融，不能给出“Python独占多少百分比”的已证实归因。
少量早期查询库内全部文档合法时可用摘要，不代表典型长期归档查询已显著省扫描。
本次唯一明显省扫描的查询就是左边界前缀；因此同时报告按文档加权比例，避免小N边界把平均省读夸大。
'''
    full_np=next(r for r in full['summary'] if r['method']=='numpy_exact')
    full_avl=next(r for r in full['summary'] if r['method']=='citra_avl' and r['epsilon']==.1)
    system_text+=f'''
完整2020–2023回放库（含相对旧cutoff的未来记录，但答案严格过滤）NumPy/AVL(.1)的P50为
{ms(full_np['timings']['end_to_end_seconds']['p50'])}/{ms(full_avl['timings']['end_to_end_seconds']['p50'])}ms。
该更慢的树结果保留在[原始回放]({link('system/summary.json')})，最终结论以不含未来记录的前缀测量为主，避免不公平夸大AVL缺点。

## 离线构建、维护与内存

24个历史前缀状态分别离线建库再重复查询，不把逐cutoff建库隐藏在热查询之外后声称总成本为零。
索引构建范围{min(x['index_build_seconds'] for x in system['prefix_builds']):.4f}–{max(x['index_build_seconds'] for x in system['prefix_builds']):.4f}秒，
合计{sum(x['index_build_seconds'] for x in system['prefix_builds']):.3f}秒；这是诊断各历史状态的成本，不是已优化的生产增量部署吞吐量。
100条真实档案的发行时间假设到达序重放，维护总耗时{maint['total_seconds']:.4f}秒，P50/P95={maint['p50_seconds']*1000:.3f}/{maint['p95_seconds']*1000:.3f}ms；
纯索引部分{maint['index_only_seconds']:.4f}秒，前缀初始化另外计费。冻结文档向量已提供，不含持续MiniLM/模型更新成本；
该保留数据没有可靠实际修订到达样本，不能把追加维护成本冒充修订传播的完整实测。
'''
    system_text+=table(['内存指标','MB (10^6 bytes)','范围'],[
        ['最大前缀索引resident估算',f"{system['memory']['index_resident_estimate']/1e6:.3f}",'Python/NumPy去重递归估算'],
        ['完整语料编码Key/Impact数组',f"{system['memory']['encoded_numpy_bytes']/1e6:.3f}",'共享表示；不含原文/基础384维输入'],
        ['主测量进程RSS',f"{system['memory']['benchmark_rss']/1e6:.3f}",'含数据、模型、解释器、各方法缓存和索引'],
        ['主测量进程峰值RSS',f"{system['memory']['benchmark_peak_rss']/1e6:.3f}",'整个共享测量进程，不能当作单方法峰值'],
        ['含维护scratch的进程峰值',f"{system['memory']['peak_including_scratch_maintenance']/1e6:.3f}",'额外诊断副本，非单个生产索引'],
        ['CUDA allocator峰值allocated/reserved',f"{system['memory']['cuda_peak_allocated']/1e6:.3f} / {system['memory']['cuda_peak_reserved']/1e6:.3f}",'不含完整驱动/context内存']])
    system_text+=f'''
没有在在线计时时开启tracemalloc来不对称拖慢Python树。RSS阶段值和索引resident定义不同，不把二者混为“Exact内存与AVL内存”的直接差。
[维护逐次记录]({link('system_asof/maintenance.json')})与主摘要保存构建/转换/cache/模型保护分项。

## 受控规模检查（明确为synthetic）

从冻结模型的2020 Key/Impact联合均值/协方差生成连续联合高斯样本，Key重新归一化；不复制相同向量冒充真实分布。
query取固定2021真实历史，预测修正W使用同一实际参数。
active_only为全部N合法、时间同质的乐观场景；mixed_archive含N合法、约N过期/未来记录和5%旧修订。
每个N/场景独立进程，现有AVL逐条构建，无Hybrid/C++/CUDA重写。每法3个query×3次热计时；冷调用保留。
规模实验每法仅9个热样本，P95/P99为这9次的经验分位数，不是生产尾延迟保证。
这里计“时间过滤+聚合+实际线性修正”，query表示预先给定，不是完整神经预测的端到端速度。
高斯矩和人为版本/时间分布不能代表真实天气归档或真实修订机制，mixed也不是实证的最坏/最好界。
'''
    scale_rows=[]
    for s in sorted(scales,key=lambda r:(r['scenario'],r['legal_n'])):
        exact=next(x for x in s['summary'] if x['method']=='numpy_exact')
        fast=next(x for x in s['summary'] if x['method']=='citra_avl' and x['epsilon']==.1)
        scale_rows.append([s['scenario'],s['legal_n'],s['physical_records'],ms(exact['total']['p50']),ms(fast['total']['p50']),
            f"{fast['scan']['mean']:.1%}",f"{s['construction_seconds']:.3f}",f"{s['memory']['index_resident_bytes']/1e6:.2f}",f"{s['memory']['process_peak_rss']/1e6:.2f}"])
    system_text+=table(['合成场景','合法N','物理版本数','Exact P50(ms)','AVL(.1) P50(ms)','扫描比','建树秒','索引MB','进程峰值MB'],scale_rows)
    system_text+='\n峰值RSS是整个worker的峰值，含样本生成、构建、查询和递归resident估算的临时集合；不是某个算法独占的纯查询峰值。\n'
    expected=len(cfg['scaling']['sizes'])*len(cfg['scaling']['scenarios'])
    system_text+=f'\n完成{len(scales)}/{expected}个注册规模/场景组合。全部ε=0/.01/.1/1的P50/P95/P99/Max、实际误差、扫描、维护、峰值RSS和原始逐次JSON保存在[scaling目录]({link("scaling/")})。\n'
    homogeneous=sorted([s for s in scales if s['scenario']=='active_only'],key=lambda s:s['legal_n'])
    wins=[]
    for s in homogeneous:
        a=next(x for x in s['summary'] if x['method']=='numpy_exact')['total']['p50']
        b=next(x for x in s['summary'] if x['method']=='citra_avl' and x['epsilon']==.1)['total']['p50']
        if b<a:wins.append((s,a,b))
    if wins:
        first,_,_=wins[0];last,a,b=wins[-1]
        lower_bound=int(np.ceil(last['construction_seconds']/(a-b)))
        system_text+=f'\n在已测网格上，active_only的第一个ε=.1优势出现在N={first["legal_n"]}（不等于精确定位了连续交叉点）；N={last["legal_n"]}时约{a/b:.1f}倍。仅摊销该次{last["construction_seconds"]:.2f}秒构建就至少需要约{lower_bound:,}次保持该优势的查询，还未计维护、内存、时间窗口失效和神经query成本。mixed_archive在已测规模没有此优势，不能把同质场景外推为真实部署加速。\n'
    system_text+=f'''
![合成规模曲线]({link('figures/synthetic_scaling_latency.png')})
![合成建树及内存]({link('figures/synthetic_build_memory.png')})

即使时间同质场景出现交叉点，也不能改写System No-Go：真实单站七天只有42篇，没有可信应用需要把此任务扩成数千/数万篇合法同质文本。
mixed归档与同质场景的差异表明交叉点依赖时态分布；合成优势不等于可覆盖真实建库/更新/内存成本。
真实场景Exact本已更快，索引没有可摊销的在线节省。

复现：见最终决策报告的命令；主逐次计时、完整回放、前缀库和合成结果全部保留，没有用Python Full Scan充当唯一Exact。
'''
    (docs/'citra_final_system_diagnosis.md').write_text(system_text)

    names=['ts_only','real_text','metadata_only','content_only','constant_text','null_no_metadata','shuffle_fixed_weights','shuffle_retrained']
    years=sorted({g['year'] for g in semantic['groups']})
    sem_text=f'''# CITRA 最终文本信息价值诊断（2026-10-29）

**Prediction: Evidence Insufficient / No-Go for the independent semantic claim。**
真实文本的总残差模型有小幅点估计改善，但不能稳定超过无语义的元数据、常量和打乱控制。
不能把相对TS的改善直接命名为文本语义收益。

## 固定协议和输入控制

所有新增方案在任何新结果前登记：同一seed7/17/29预训练TS、完全冻结骨干、文本头零初始化、
Window7 Count Mean、48→6小时、相同2020训练和2021最佳epoch验证、AdamW/20epoch/batch32/lr=.001/梯度限制1。
原真实文本和重训打乱的各3份checkpoint复用，固定权重打乱不重训。
新增Metadata-only、Content-only、Constant Text、Null/no metadata共12个训练。
内容消融的文本残差模型均20,620参数，骨干1,164固定，19,456可训练；TS-only只执行该骨干。
常量输入会缩小有效函数空间、部分输入权重无梯度，
这是控制本身的必要差异，不能把相同分配参数量说成完全相同有效容量。

|方案|实际输入|
|---|---|
|真实文本|原冻结MiniLM384维 + 发行月份sin/cos|
|Metadata-only|384维全0；只保留发行月份sin/cos，无文本/embedding访问|
|Content-only|原MiniLM；Impact Encoder的两项月份元数据固定0|
|Constant Text|所有文档同一固定单位基向量；保留月份sin/cos|
|Null/no metadata|所有embedding和月份都0，仍用同一历史条件query与残差网络|
|固定权重因果打乱|真实模型参数不变，内容从时槽激活前合法供体中前缀有放回采样|
|同容量重训打乱|同样的因果供体规则，使用已有同预算重训权重|
|Text-off / TS-only|完全关闭修正，每种种子的数值预测与原TS逐元素相同|

没有加入未登记的公开时间位置/N特征。合法集合和计数归一分母本身保留时间/集合结构；
常量Key/Impact在非空窗口的计数归一会抵消N，因此Null主要是历史数值条件的额外校准函数。
Query Encoder仍可读取历史数值、已知目标与horizon；不能把由此产生的季节信息称作“纯文本”。
Content-only只删除直接月份sin/cos，MiniLM本身可能编码原始发行头/文本中的日期或季节词，未宣称消除了所有隐式季节信息。
非内容控制通过字段白名单构造输入，测试会使控制构造阶段任何原embedding/text访问抛错；不会把真实内容送入网络或用其统计量生成非内容输入。
“无内容访问”指该特征/模型路径，不声称来源审计和原始JSON加载在IO层面从未读取包含内容的文件字节；来源fingerprint只绑定产物，不作为模型特征或随机种子。
因果替换供体均不晚于原时槽激活，原时槽在查询截止前合法才可参与；没有全局未来池置换。

## 年份角色与新独立检验

2020训练、2021验证；2022/2023此前已查看，本轮只做解释性诊断。
本轮注册后下载2024，同站/办公室、相同协议，全部方法权重冻结后才计算2024预测指标。
2024没有调窗口、网络、超参、表示或挑选种子；这是新年份验证，但仍只有一个地区，也不是获证历史到达回测。
2024原始AFD2228份，5份发行头冲突隔离，保留2223份；ASOS8784行，无重复时间组，风速缺失3格；
旧数值、时间可用性、原文身份和旧向量前缀保持，Scaler仅2020拟合。各年共同起点数为363/364/366。
[注册/环境]({link('registration.json')})、[新年份provenance]({link('new_year_manifest.json')})、
[全部方法冻结记录]({link('semantics/methods_frozen.json')})。

## 全部种子与逐目标结果

以下标准化MSE为三种训练种子均值±样本标准差，不能将同一天气目标当作3份独立观测。
'''
    sem_text+=table(['方案']+[str(y) for y in years],[[n]+[meanstd(next(g for g in semantic['groups'] if g['kind']==n and g['year']==y)['metrics']['normalized_mse']) for y in years] for n in names])
    sem_text+='\n标准化MAE：\n'+table(['方案']+[str(y) for y in years],[[n]+[meanstd(next(g for g in semantic['groups'] if g['kind']==n and g['year']==y)['metrics']['normalized_mae']) for y in years] for n in names])
    sem_text+='\n2024每一种子标准化MSE（不只展示最优种子）：\n'+table(['方案','seed7','seed17','seed29'],[[n]+[f'{v:.9f}' for v in next(g for g in semantic['groups'] if g['kind']==n and g['year']==2024)['metrics']['normalized_mse']['values']] for n in names])
    for year in years:
        sem_text+=f'\n{year}逐目标物理单位指标（均值±种子标准差）：\n'
        sem_text+=table(['方案','温度MSE °F²','温度MAE °F','风速MSE knot²','风速MAE knot'],[[n]+[meanstd(next(g for g in semantic['groups'] if g['kind']==n and g['year']==year)['metrics']['per_target'][t][m]) for t,m in (('temperature_F','mse'),('temperature_F','mae'),('wind_speed_knots','mse'),('wind_speed_knots','mae'))] for n in names])
    sem_text+=f'''
逐种子全部目标指标在[per_seed_metrics.csv]({link('semantics/per_seed_metrics.csv')})，
逐预测日期MSE/MAE、物理偏差、修正L2和数值L2在[daily_losses.csv]({link('semantics/daily_losses.csv')})，
原预测、truth、cutoff、门控和N在各模型目录NPZ；新训练曲线保存在原fit日志，没有只保存最佳一次。
修正幅度和偏差汇总在[summary.json]({link('semantics/summary.json')})，所有Text-off数组均通过与固定TS的逐元素核对。

![全部模型跨年诊断]({link('figures/semantic_controls.png')})

## 配对时间相关性分析

预注册固定7/14**日历日**循环移动块、2000次、固定种子54321+年份。
先在每个日期对三种训练种子的配对损失差求平均，再沿日历块重采样；缺失日期保留NaN并在块均值中忽略。
没有把3×366行当作1098份独立天气。下面Δ=真实文本−对照，负值才有利于真实文本。
CI是未做多重比较校正的percentile区间，条件于这3份已训练模型；种子不确定性另列标准差，
块长不能排除所有更长时序相关、季节非平稳性和单地区偏差，不能解释为地理泛化置信区间。
'''
    for year in years:
        sem_text+=f'\n{year}标准化配对MSE：\n'
        rs=[]
        for n in names:
            if n=='real_text':continue
            a=next(r for r in semantic['paired'] if r['year']==year and r['metric']=='normalized_mse' and r['contrast']=='real_text minus '+n and r['block_days']==7)
            b=next(r for r in semantic['paired'] if r['year']==year and r['metric']=='normalized_mse' and r['contrast']=='real_text minus '+n and r['block_days']==14)
            rs.append([n,f"{a['mean_difference']:+.7f}",f"[{a['ci95_low']:+.7f}, {a['ci95_high']:+.7f}]",f"[{b['ci95_low']:+.7f}, {b['ci95_high']:+.7f}]",f"{a['improvement_date_fraction']:.1%}"])
        sem_text+=table(['对照','ΔMSE','7天块95%CI','14天块95%CI','改善日期比例'],rs)
    sem_text+=f'''
全部标准化/温度/风速MSE和MAE的配对区间见[paired_bootstrap.json]({link('semantics/paired_bootstrap.json')})。

![块Bootstrap配对区间]({link('figures/paired_block_intervals.png')})

## 结论与不可归因部分

2024真实文本MSE约0.308067，Metadata-only约0.307816，后者无需任何语义输入就有更好的点估计。
Constant/Null和重训打乱均可获得部分总残差改善；固定权重打乱仍接近真实文本。
2024真实文本相对于TS、元数据、内容去月份、常量、Null和两种打乱的标准化联合MSE之7/14天CI全部跨0。
不能把这句话推广成所有目标都无差异：气温MSE相对Metadata-only的7天块Δ约+0.1363°F²，
未校正95%CI约[+0.0342,+0.2472]，方向反而不利于真实文本；部分气温MAE对照也不利。
这不是等价性证明，也不证明所有文本模型无用，而是没有达到“稳定独立语义贡献”的证据标准。
2022真实文本弱于TS和Metadata-only，2023较好，2024仍未超过无内容元数据，跨年结论不稳定。
没有选择表现更好的Metadata-only代替原参照后重新包装为CITRA语义成功。

新增12个训练只用于最后的内容归因；未新增大型网络、未修改旧预测结果、没有看2024后再训练。
[训练/评价/统计成本文件]({link('semantics/')})保留墙钟；原真实/打乱/TS的既有预训练不是“免费新训练”。
因此Prediction判断为Evidence Insufficient，当前CITRA的文本价值主张No-Go。
'''
    (docs/'citra_final_semantic_diagnosis.md').write_text(sem_text)

    decision=f'''# CITRA 最终Go/No-Go决策（2026-10-29）

**最终建议：No-Go for Current CITRA。停止当前论文主线，不开发Hybrid或新复杂索引。**

## 原始研究主张与已验证部分

原主张是在Point-in-Time文档可见性/版本规则下，用层次矩摘要和泰勒余项近似完整合法核和，
将误差传到同一冻结预测函数的输出，并同时获得有意义的预测收益和系统效率。
已验证：独立snapshot/oracle与版本区间语义、未知可用时间排除、合法窗口/计数归一、
固定参数的解析聚合余项、实际线性头/门控范数传播、参数/语料变更失效检查，以及真实可训练的预测路径。
这些是可复用的正确性/工程成果；不是形式化IEEE总舍入认证，也不自动构成ICDE方法创新。

## 两个维度独立决策

|维度|决策|实际证据|
|---|---|---|
|System|No-Go|真实七天P50=42篇、P95=44、Max=47；强Exact已亚毫秒；as-of前缀AVL(.1)仍更慢|
|Prediction|Evidence Insufficient / semantic claim No-Go|新2024真实文本不优于Metadata-only点估计；主指标配对CI跨0，部分气温指标更差；旧年份不稳定|
|论文路线|No-Go for Current CITRA|系统必要性和独立文本信息价值均未达到证据门槛|

System的同参数前缀库NumPy/AVL(.1)端到端P50={ms(npref['timings']['end_to_end_seconds']['p50'])}/{ms(eps01['timings']['end_to_end_seconds']['p50'])}ms，
纯向量化聚合只需{ms(npref['timings']['aggregation_seconds']['p50'])}ms；没有可信真实工作负载支撑把当前42篇扩成数千/数万篇。
合成同质场景即使有交叉点，也不抵消真实查询更慢、构建/维护/内存额外成本或混合归档退化。

Prediction在2024的三种子均值：TS0.309695、真实0.308067、Metadata-only0.307816、
Content-only0.308539、Constant0.308377、Null0.308837、同权重打乱0.308240、重训打乱0.308369。
不可把相对TS约0.53%的总残差改善称为语义贡献；其7/14日块CI连相对TS的改善也未稳健排除0。
2023此前已看过，本轮不把它再称为独立测试。2024是锁定后的新年份，但仍只有同一地区，
首次历史公开时间没有获证，始终明确为issuance_asof_assumption。

## 值得保存的成果与停止边界

- Point-in-Time权威版本选择、可见区间推导、未知/模拟/verified的显式区分。
- 独立Exact Oracle、float64容差测试、解析余项与冻结模型输出预算的绑定。
- Temporal AVL的不可变节点/局部维护原型与完整负性能记录。
- Count Mean/七天窗口算子、固定TS残差和输入白名单归因控制。
- 连续真实原文/观测、SHA、冻结MiniLM身份绑定、所有种子/预测/配对日期与再现命令。
- 避免种子伪重复的日历块Bootstrap、严格区分旧诊断年份和新锁定验证。

停止当前主线不等于否认这些组件可在别处复用；本轮没有可信新应用场景，故不使用空泛Conditional Go延长项目。
不自动进入Hybrid、模型扩容、C++/CUDA重写或新数据结构开发。原ReTaMTSF路径和全部历史产物不改，不推送远端。

## 产物和复现

系统证据：[citra_final_system_diagnosis.md](citra_final_system_diagnosis.md)。
语义证据：[citra_final_semantic_diagnosis.md](citra_final_semantic_diagnosis.md)。
原始注册、环境/commit/checkpoint/source SHA、逐次性能、模型、预测、日期损失、Bootstrap、图、异常和测试在
[{root}]({os.path.relpath(root,docs)}/)。完整协议为`configs/citra_final.yaml`。

本机Python3.10、NumPy2.2.5、Torch2.4.1/CUDA12.1/A100，Torch/NumPy CPU线程固定1。
因已有iJIT环境问题使用此前官方ITT兼容实现：
`LD_PRELOAD=/tmp/citra-ittapi/libjitprofiling.so`。正常Torch环境不需要该步骤。
绘图使用可选`requirements-citra-diagnosis.txt`，语义基础编码用`requirements-citra-text.txt`。
首次运行以下命令前，先按上轮天气报告及本轮commands文件准备被Git忽略的数据；
已有输出不会强制覆盖，重跑应复制协议并将output改到新目录：

```bash
python - <<'PY'
import yaml
from pathlib import Path
c = yaml.safe_load(Path('configs/citra_final.yaml').read_text())
c['output'] = '/tmp/citra-final-rerun'
Path('/tmp/citra-final-rerun.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
PY
export CITRA_FINAL_CONFIG=/tmp/citra-final-rerun.yaml
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/register_citra_final.py
python scripts/benchmark_citra_final.py
python scripts/benchmark_citra_final.py --archive-mode asof_prefix
python scripts/run_citra_final_semantics.py train --device cuda:0
python scripts/run_citra_final_semantics.py evaluate --device cuda:0 --include-new
python scripts/run_citra_final_semantics.py analyze
python scripts/benchmark_citra_scaling.py prepare
python scripts/benchmark_citra_scaling.py all
python scripts/report_citra_final.py --docs-out /tmp/citra-final-reports
python scripts/verify_citra_final.py
python -m unittest discover -s tests -v
```

上面复制协议只更换产物位置，不改变冻结方法或种子；报告依赖原始年份数据和既有checkpoint。
新年份采集、单独编码/追加及实际带环境的执行命令见[commands.md]({link('commands.md')})。
失败/异常在[failures.jsonl]({link('failures.jsonl')})，没有删除失败种子或把合成曲线写成真实应用证据。
实际最终回归83项通过，其中新增8项覆盖非内容输入不可访问、直接月份清零、数值/合法集合不变、
缓存筛选与版本Oracle、种子不扩大天气样本量、块相关性/日期缺失及重复/跨年拒绝；
见[tests_final.txt]({link('tests_final.txt')})。核心模型、原独立Oracle和AVL实现与起始commit保持一致，
未修改原ReTaMTSF路径；新增代码是诊断、控制、统计和报告脚本。
独立核对21份checkpoint、20个原始/元数据文件、72组预测、42个主指标配对区间、
3374次真实性能调用以及10个合成场景的500次查询通过，见[verification.json]({link('verification.json')})。
训练时源码另有SHA匹配快照；后加的fresh-output配置入口不改变已登记的输入控制、统计方法或模型数学。
'''
    (docs/'citra_final_go_nogo.md').write_text(decision)


def seed_metrics(root):
    records=read(root/'semantics/evaluations.json')['rows'];out=[]
    for r in records:
        m=r['text_on'];scale=np.asarray(r['scaler']['scale'])
        out.append(dict(kind=r['kind'],year=r['year'],seed=r['seed'],windows=r['windows'],normalized_mse=m['normalized_mse'],normalized_mae=m['normalized_mae'],
            temperature_mse=m['per_target']['temperature_F']['mse'],temperature_mae=m['per_target']['temperature_F']['mae'],
            wind_mse=m['per_target']['wind_speed_knots']['mse'],wind_mae=m['per_target']['wind_speed_knots']['mae'],
            temperature_bias=m['normalized_bias_by_target'][0]*scale[0],wind_bias=m['normalized_bias_by_target'][1]*scale[1]))
    with (root/'semantics/per_seed_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)


def scaling_csv(root,scales):
    rows=[]
    for s in scales:
        for r in s['summary']:
            rows.append(dict(synthetic=True,scenario=s['scenario'],legal_n=s['legal_n'],physical_records=s['physical_records'],
                method=r['method'],epsilon=r['epsilon'],**{'total_'+k:r['total'][k] for k in ('p50','p95','p99','max')},
                aggregation_p50=r['aggregation']['p50'],scan_mean=r['scan']['mean'],construction_seconds=s['construction_seconds'],
                index_resident_bytes=s['memory']['index_resident_bytes'],process_peak_rss=s['memory']['process_peak_rss']))
    if rows:
        with (root/'scaling/summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--docs-out',type=Path,default=Path('docs'));a=p.parse_args();a.docs_out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(protocol_config_path().read_text());root=Path(cfg['output'])
    counts=read(root/'system/document_counts.json');system=read(root/'system_asof/summary.json');full=read(root/'system/summary.json');semantic=read(root/'semantics/summary.json')
    scales=[read(p) for p in (root/'scaling').glob('*.json') if p.name!='basis.json']
    seed_metrics(root);scaling_csv(root,scales);figures(root,counts,system,semantic,scales);reports(cfg,root,a.docs_out,counts,system,full,semantic,scales)
