"""Build the self-contained teaching notebooks from audited project sources."""
import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent

def source(path):
    nb = json.loads((ROOT / path).read_text(encoding='utf-8'))
    return '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')

def md(s):
    return {'cell_type':'markdown','metadata':{},'source':s.strip()}

def code(s):
    s=textwrap.dedent(s).strip()
    ast.parse(s)
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s}

def pair(cells,title,explanation,s):
    cells.extend([md('## '+title+'\n\n'+explanation),code(s)])

def save(name,cells):
    # 复杂块附带逐句定位表，教程可在不改变历史代码作用域的情况下解释内部函数。
    expanded=[]
    for cell in cells:
        expanded.append(cell)
        if cell['cell_type']=='code' and 'def ' in cell['source']:
            explanations=[]
            rules=[('np.bincount','按整数分组编号累计人数；weights指定时累计对应标签，得到购买人数。'),
                   ('np.convolve','把每个收入箱及左右邻箱按核权重汇总，得到更稳定的局部统计。'),
                   ('np.searchsorted','按已学习的边界把数值映射为分组编号；right决定恰好位于边界时的归属。'),
                   ('np.linspace','在训练收入最小值与最大值之间建立等宽边界。'),
                   ('np.triu','只检查相关矩阵上三角，删除后出现的重复信息列，保留之前的列顺序。'),
                   ('fit_transform','训练行使用编码器内部交叉拟合结果，不能替换成fit后对训练集transform。'),
                   ('encoder.transform','使用外层训练数据已学到的映射，不输入验证或测试标签。'),
                   ('value_counts','从当前训练部分计算频率；normalize=True返回比例，否则返回次数。'),
                   ('pd.concat','按列合并特征；索引必须一致，reset_index用于避免错位产生缺失值。'),
                   ('np.column_stack','把等长数组并排组成二维特征矩阵，顺序与列名列表对应。'),
                   ('pd.Categorical','用训练折词表固定类别编码；未见类别成为缺失，不重新为验证集编号。'),
                   ('np.rint','四舍五入到最近整数；与floor向下取整含义不同。'),
                   ('np.floor','向下取整建立分组；输入除以100或1000时表示更粗的收入区间。'),
                   ('np.flatnonzero','把布尔条件中为真的位置取出，作为iloc的行位置。'),
                   ('fillna','为未匹配或缺失项提供明确回退值；均值特征通常回退整体购买比例。'),
                   ('astype','固定数据类型；float32节约内存，int类型表达离散分组，str把数值作为类别键。')]
            for number,line in enumerate(cell['source'].splitlines(),1):
                for token,meaning in rules:
                    if token in line and not line.lstrip().startswith('#'):
                        explanations.append(f'| {number} | `{line.strip().replace("|"," / ")}` | {meaning} |')
                        break
            if explanations:
                expanded.append(md('### 对照上方代码逐句理解\n\n行号从上方代码单元第一行起计。重复操作也列出，方便逐行定位；跨行调用请连同后续参数一起阅读。\n\n| 行 | 代码定位 | 中文解释 |\n|---|---|---|\n'+'\n'.join(explanations)))
    cells=expanded
    for index, cell in enumerate(cells):
        cell['id'] = f'lesson-{index:03d}'
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    (DEST/name).write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')

COMMON = '''
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json
import gc
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from IPython.display import display
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold

INPUT = Path('/kaggle/input')
TARGET = 'Will_Buy_EV'
SEED = 42
N_SPLITS = 5

def unique_file(name):
    """从已挂载输入中定位唯一文件；多个版本时停止，防止静默读错。"""
    paths = list(INPUT.rglob(name))
    assert len(paths) == 1, f'Expected one {name}, found {paths}'
    return paths[0]

competition_dirs = [INPUT/'competitions/playground-series-s6e9', INPUT/'playground-series-s6e9']
available = [p for p in competition_dirs if (p/'train.csv').is_file()]
assert len(available) == 1, 'Attach the official competition data.'
DATA = available[0]
train = pd.read_csv(DATA/'train.csv')
test = pd.read_csv(DATA/'test.csv')
sample = pd.read_csv(DATA/'sample_submission.csv')
y = train[TARGET].map({'No':0, 'Yes':1})
assert y.notna().all() and set(y.unique()) == {0,1}
assert train.id.is_unique and test.id.is_unique
assert sample.columns.tolist() == ['id',TARGET] and sample.id.equals(test.id)
assert train.columns.drop(['id',TARGET]).tolist() == test.columns.drop('id').tolist()

def align_rows(frame, ids):
    """先检查一一对应，再按官方顺序排列；不能直接假设CSV行序相同。"""
    assert frame.id.is_unique and len(frame) == len(ids)
    assert set(frame.id) == set(ids)
    return frame.set_index('id').loc[ids].reset_index()

def current_versions():
    return {'lightgbm':lgb.__version__, 'sklearn':sklearn.__version__,
            'numpy':np.__version__, 'pandas':pd.__version__}
'''

LOAD_FOLDS='''
fold_path = unique_file('shared_folds.csv')
shared = align_rows(pd.read_csv(fold_path), train.id)
assert np.array_equal(shared.target, y)
assert shared.fold.notna().all() and shared.fold.isin(range(5)).all()
assert set(shared.fold) == set(range(5))
fold_ids = shared.fold.to_numpy(dtype=int)
foundation_note = json.loads((fold_path.parent/'dataset_note.json').read_text())
display(shared.groupby('fold').agg(rows=('id','size'),positive_rate=('target','mean')))
print(current_versions())
'''

DOCS='''
- [比赛数据与规则](https://www.kaggle.com/competitions/playground-series-s6e9)
- [LightGBM论文](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [目标编码：内部交叉拟合与平滑](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
- [AUC定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
'''

def intro(title,owner,inputs,method):
    return [md(f'# {title}\n\n负责人：{owner}。这是中文教学参考，正式实现由成员理解后编写、执行并核对。\n\n输入挂载：{inputs}。无需挂载旧track代码包。CPU训练，四线程；机器等待另计。\n\n{method}\n\n'+DOCS+'\n本教程生成时尚未执行Kaggle完整训练。历史分数是核对参照，不是本轮结果。阅读当前文档不意味着升级历史环境。')]

def guide(cells):
    cells.append(md('## 如何学习本文件\n\n每次只运行一个单元，先用自己的话预测输出。`iloc`按位置取行，`loc`按标签取行；`to_numpy`去掉索引，之后必须保证位置对应。`assert`是验收条件，失败应查数据而非删除检查。`fit`从数据学习，`transform`使用已学习规则。\n\n编程练习：修改一个小例子的输入并解释变化；正式配置保持历史定义。复杂特征组的整体增益不能归因于单一列。'))

cells=intro('数据检查与共享五折验证','A主实现，B/C复核','官方比赛数据，以及历史收入邻域模型和混合特征模型的已保存输出','从已有OOF恢复历史分组，避免重新随机划分造成不可比较。历史预测仅用于提取分组和后续核对，新的模型会重新训练。')
guide(cells)
pair(cells,'读取数据与定义对齐函数','把行编号作为连接键；标签映射必须完整。输出应显示读取成功，没有目标缺失。',COMMON)
pair(cells,'明确历史文件路径','下面路径来自已运行的最终方案。若Kaggle挂载路径不同，只修改这两行，指向包含OOF、submission和run_summary的目录。必须先保存原Notebook输出并Add Input；上传ipynb不会附带CSV。', '''
history_neighborhood = INPUT/'notebooks/nemonade/10-multiscale-encoding/R003_multiscale_encoding_20260909_071033_059889'
history_hybrid = INPUT/'notebooks/nemonade/22-hyl001-hybrid-lightgbm/HYL001_hybrid_lightgbm_20260911_153854_073213'
history = {}
for name, directory in [('neighborhood',history_neighborhood),('hybrid',history_hybrid)]:
    assert directory.is_dir(), f'Attach saved output: {directory}'
    oof = align_rows(pd.read_csv(directory/'oof_predictions.csv'),train.id)
    assert np.array_equal(oof.target,y)
    assert oof.fold.isin(range(5)).all() and set(oof.fold)==set(range(5))
    pred_col = 'probability' if 'probability' in oof else 'prediction'
    assert oof[pred_col].between(0,1).all()
    summary = json.loads((directory/'run_summary.json').read_text())
    history[name] = {'directory':str(directory),'oof':oof,'summary':summary}
    print(name, roc_auc_score(y,oof[pred_col]))
assert np.array_equal(history['neighborhood']['oof'].fold,history['hybrid']['oof'].fold)
shared = history['neighborhood']['oof'][['id','target','fold']].copy()
display(shared.groupby('fold').agg(rows=('id','size'),positive_rate=('target','mean')))
''')
pair(cells,'冻结共享文件','输出只包含分组和数据说明。没有模型预测、代码哈希或提交名单。环境版本来自两次历史运行，后续训练前逐项核对。', '''
output = Path('/kaggle/working/team_foundation')
output.mkdir(exist_ok=False)
shared.to_csv(output/'shared_folds.csv',index=False)
note = {'target_mapping':{'No':0,'Yes':1}, 'train_rows':len(train),'test_rows':len(test),
        'feature_columns':test.columns.drop('id').tolist(),
        'fold_source':history['neighborhood']['directory'],
        'historical_versions':{k:v['summary'].get('versions',{}) for k,v in history.items()},
        'historical_features':{k:v['summary'].get('fold_features',{}) for k,v in history.items()},
        'historical_params':{k:v['summary'].get('model_params',{}) for k,v in history.items()},
        'history_directories':{k:v['directory'] for k,v in history.items()},
        'versions_when_preparing':current_versions()}
(output/'dataset_note.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
print(output)
''')
cells.append(md('## 交接与理解检查\n\n将`team_foundation`中的两个文件保存为一个固定版本Kaggle Dataset，三人挂载同一版本。已存在输出目录时先保存当前运行，再重开会话，避免覆盖。\n\nB检查每行标签和fold；C故意打乱共享文件行序后用`align_rows`恢复并验证一致。问题：相同seed为什么不保证不同原始行序下分组相同？历史OOF为什么不能当成本轮新训练结果？'))
save('00_data_and_shared_validation.ipynb',cells)

# 原始函数完整内嵌，机械重命名只改善可读性，不改变计算顺序。
helper=(ROOT/'notebooks/r002_features.py').read_text(encoding='utf-8')
helper=helper[helper.index('def make_r002_preparer'):]
for old,new in [('make_r002_preparer','make_neighborhood_preparer'),('build_r001_features','build_public_features'),('prepare_r001_fold','prepare_public_fold')]:
    helper=helper.replace(old,new)
helper=helper.replace('def prepare_fold(train_idx, valid_idx, expected_columns):','def prepare_fold(train_idx, valid_idx):').replace('            assert frame.columns.tolist() == expected_columns\n','')
helper=helper.replace('# Match the public recipe\'s numerical digit extraction.','# 数字位提取保持原公式；浮点整除不可随意替换为四舍五入。').replace('# Learn frequency mappings from the training fold only.','# 频率只用外层训练折；未知类别频率填0。').replace('# Select features without inspecting validation or test distributions.','# 仅在训练折识别常数与完全相关列，保持列顺序和精确比较。').replace('# Cross-fit both encoding views inside the outer training fold.','# 两种平滑强度分别内部交叉拟合；验证和测试只transform。')
helper=helper.replace('        counts = np.bincount', '        # bincount统计每个收入区间人数；带weights时统计正例总数。\n        counts = np.bincount').replace('        center = (totals', '        # 平滑：增加smooth个按整体比例分布的虚拟样本。\n        center = (totals').replace('        kernel = np.exp','        # 高斯核让相邻区间贡献较小权重；convolve汇总左右邻居。\n        kernel = np.exp').replace('            train_features = np.empty','            # 空数组按内部留出行回填，避免该行标签直接参与自身编码。\n            train_features = np.empty')

hybrid=source('notebooks/22-hyl001-hybrid-lightgbm.ipynb')
hybrid_features=hybrid[hybrid.index('feature_columns ='):hybrid.index('model_params =')]
hybrid_features=hybrid_features.replace('def prepare_fold(training_index, validation_index):','def prepare_hybrid_fold(training_index, validation_index):')
hybrid_features=hybrid_features.replace('    income = np.floor','    # 纯数值规则不学习比赛标签；floor与rint的差别必须保留。\n    income = np.floor').replace('    for smooth, tag','    # 三种平滑提供不同分组统计视角；训练行使用内部交叉拟合。\n    for smooth, tag').replace('        categories = pd.Index','        # 类别词表只从外层训练折学习，未知值成为缺失类别。\n        categories = pd.Index')
hybrid_params=ast.literal_eval(ast.parse(hybrid[hybrid.index('model_params ='):hybrid.index('candidate_oof =')]).body[0].value) if False else None
rparams=json.loads((ROOT/'outputs/Notebook7/R002/run_summary.json').read_text())['model_params']
hparams_code=hybrid[hybrid.index('model_params ='):hybrid.index('candidate_oof =')]

TRAIN='''
candidate_oof = np.full(len(train),np.nan)  # 没有预测的行保持NaN，便于发现漏填。
candidate_test = np.zeros(len(test))
coverage = np.zeros(len(train),dtype=np.uint8)
records, fold_features = [], {}
started = perf_counter()
for fold in range(N_SPLITS):
    training_index = np.flatnonzero(fold_ids != fold)
    validation_index = np.flatnonzero(fold_ids == fold)
    X_train, X_valid, X_test = prepare_fold(training_index,validation_index)
    assert X_train.columns.equals(X_valid.columns) and X_train.columns.equals(X_test.columns)
    if expected_count is not None:
        assert X_train.shape[1] == expected_count, (fold,X_train.shape)
        historical_columns = foundation_note.get('historical_features',{}).get(run_name,{}).get(str(fold))
        if historical_columns is not None:
            assert X_train.columns.tolist() == historical_columns, 'Historical feature order differs.'
    fold_features[str(fold)] = X_train.columns.tolist()
    if run_name != 'baseline':
        assert model_params == foundation_note['historical_params'][run_name], 'Historical parameters differ.'
    model = lgb.LGBMClassifier(**model_params)  # 每个外层fold创建全新模型。
    fold_started = perf_counter()
    model.fit(X_train,y.iloc[training_index],eval_set=[(X_valid,y.iloc[validation_index])],
              eval_metric='auc',callbacks=[lgb.early_stopping(patience,first_metric_only=True,verbose=False),lgb.log_evaluation(1000)])
    probability = model.predict_proba(X_valid,num_iteration=model.best_iteration_)[:,1]
    test_probability = model.predict_proba(X_test,num_iteration=model.best_iteration_)[:,1]
    assert np.isfinite(probability).all() and np.isfinite(test_probability).all()
    candidate_oof[validation_index] = probability  # 回填官方训练行的位置。
    coverage[validation_index] += 1
    candidate_test += test_probability/N_SPLITS  # 在概率空间平均，暂不排名。
    records.append({'fold':fold,'auc':float(roc_auc_score(y.iloc[validation_index],probability)),
                    'features':X_train.shape[1],'best_iteration':int(model.best_iteration_),
                    'fit_seconds':perf_counter()-fold_started})
    print(records[-1])
    del model,X_train,X_valid,X_test
    gc.collect()
assert (coverage==1).all() and np.isfinite(candidate_oof).all()
assert ((candidate_oof>=0)&(candidate_oof<=1)).all()
assert ((candidate_test>=0)&(candidate_test<=1)).all()
elapsed = perf_counter()-started
display(pd.DataFrame(records))
print('Overall OOF AUC:',roc_auc_score(y,candidate_oof))
'''
EXPORT='''
output = Path('/kaggle/working')/run_name
output.mkdir(exist_ok=False)
oof = pd.DataFrame({'id':train.id,'target':y,'fold':fold_ids,'prediction':candidate_oof})
submission = sample.copy()
submission[TARGET] = candidate_test
auc = float(roc_auc_score(y,candidate_oof))
report = (f'The {run_name} model was trained on five shared folds. Overall OOF ROC AUC was {auc:.9f}. '
          'Test probabilities were averaged across five models. All competition-derived supervised '
          'preprocessing was fitted within outer training folds. These are development results; '
          'leaderboard performance for this run remains unverified.')
summary = {'run_name':run_name,'model_params':model_params,'stopping_rounds':patience,
           'fold_features':fold_features,'fold_metrics':records,'oof_auc':auc,'elapsed_seconds':elapsed,
           'versions':current_versions(),'fold_source':str(fold_path),'method_sources':method_sources,
           'test_aggregation':'mean probabilities across five folds','public_score':None,'report_summary':report}
oof.to_csv(output/'oof_predictions.csv',index=False)
submission.to_csv(output/'submission.csv',index=False)
(output/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
print(report)
print('Saved:',output)
'''

for kind,title,owner in [('baseline','原始特征LightGBM基线','C'),('neighborhood','收入邻域与多尺度编码LightGBM','A'),('hybrid','混合特征与多组目标编码LightGBM','B')]:
    inputs='官方比赛数据、任务00输出的固定共享Dataset'+('、原始EV数据集中的EV_Adoption_and_Range_Anxiety_Dataset.csv' if kind!='baseline' else '')
    method={'baseline':'本项目教学对照：原始特征和基本类别处理。配置为明确的基线设计，不能声称逐项复现某个历史基线成绩。','neighborhood':'公开方法适配与项目收入邻域实现。来源：Naji基础配方、多尺度编码与Zoom Zoom局部统计思想；完整历史函数在本文件展开。','hybrid':'依据Megayak公开Hybrid配方及本项目已运行的五折适配。保留实际阈值与参数以复现历史方案；阈值属于经验规则，不能声称理论最优。'}[kind]
    cells=intro(title,owner,inputs,method)
    guide(cells)
    pair(cells,'读取官方数据','路径检查避免读错版本；对齐函数先检查ID集合，再恢复官方顺序。',COMMON)
    pair(cells,'读取共同分组','共享fold决定训练与验证行，三人不能重新划分。',LOAD_FOLDS)
    if kind!='baseline':
        pair(cells,'核对环境和读取原始数据','表格比较历史与当前环境。若版本不同，先在Kaggle安装历史摘要记录的版本并重启会话，再从头运行；不要静默升级。原始数据用于统计特征，训练模型仍使用比赛训练行。',f'''
        expected_versions = foundation_note['historical_versions']['{kind}']
        normalized = {{('sklearn' if k=='scikit_learn' else k):v for k,v in expected_versions.items()}}
        comparison = pd.DataFrame({{'historical':normalized,'current':current_versions()}})
        display(comparison)
        assert normalized, 'Historical environment metadata is missing.'
        assert all(current_versions().get(k)==v for k,v in normalized.items() if k in current_versions()), 'Match historical library versions and restart the session.'
        original_path = unique_file('EV_Adoption_and_Range_Anxiety_Dataset.csv')
        original = pd.read_csv(original_path)
        print(original.shape)
        ''')
        cells.append(md('### 原始EV数据的身份与保留理由\n\n[官方Data页面](https://www.kaggle.com/competitions/playground-series-s6e9/data)明确链接这份[Omkar Kadam发布的EV源数据](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety)，措辞是比赛训练/测试数据受其启发、分布接近但不相同。源数据本身是10,000行合成记录，不能当作真实调查。\n\n历史强模型删除12列外部均值后OOF仅下降约0.00000515、公开榜显示持平；不能宣称外部均值已有可靠私人榜收益。本次保留是为了重建已验证的完整配置。'))
    if kind=='baseline':
        pair(cells,'构造基础特征','只使用原始列。每折从训练部分学习类别词表；未知类别成为缺失值，由LightGBM处理。数值缺失也由模型处理，不用验证数据估计填充值。', '''
        feature_columns = test.columns.drop('id').tolist()
        categorical_columns = train[feature_columns].select_dtypes(include=['object','string','category']).columns.tolist()
        def prepare_fold(training_index,validation_index):
            """输入两组行位置；输出列一致的训练、验证和测试特征。"""
            frames = [train.iloc[training_index][feature_columns].copy(),train.iloc[validation_index][feature_columns].copy(),test[feature_columns].copy()]
            for column in categorical_columns:
                categories = pd.Index(frames[0][column].astype(str).unique())
                for frame in frames:
                    frame[column] = pd.Categorical(frame[column].astype(str),categories=categories)
            return frames
        model_params = dict(objective='binary',n_estimators=10000,learning_rate=0.05,num_leaves=31,random_state=42,n_jobs=4,verbosity=-1)
        patience, expected_count = 200, None
        ''')
    elif kind=='neighborhood':
        pair(cells,'实现基础特征与收入邻域函数','函数工厂先准备原始数据均值，再返回每折特征函数。这里只定义函数，不训练模型。内部函数按顺序完成数字位→原始数据均值→频率→删除常数/重复信息→双平滑目标编码→邻域统计。`np.bincount`统计每箱人数和购买人数；平滑均值=(购买数+10×整体比例)/(人数+10)；`convolve`汇总左右相邻区间。斜率是右减左，曲率是中心减左右均值。8192/16384两种分箱提供不同尺度。',helper)
        pair(cells,'追加四种分组的购买统计','例如收入12345分别映射到12345、123、12；粗分组样本多但分辨率低。训练编码使用fit_transform内部交叉拟合，验证和测试仅transform。smooth=100将稀疏组向整体比例收缩。', '''
        prepare_neighborhood = make_neighborhood_preparer(train,test,original)
        def prepare_fold(training_index,validation_index):
            frames = prepare_neighborhood(training_index,validation_index)
            keys = [pd.DataFrame({
                'Income_Integer':np.floor(part.Annual_Income_USD).astype('int64'),
                'Income_100':np.floor(part.Annual_Income_USD/100).astype('int64'),
                'Income_1000':np.floor(part.Annual_Income_USD/1000).astype('int64'),
                'Commute_Integer':np.floor(part.Daily_Commute_km).astype('int64')
            },index=part.index) for part in [train.iloc[training_index],train.iloc[validation_index],test]]
            encoder = TargetEncoder(target_type='binary',smooth=100.0,cv=5,shuffle=True,random_state=42)
            values = [encoder.fit_transform(keys[0],y.iloc[training_index]),encoder.transform(keys[1]),encoder.transform(keys[2])]
            for frame,encoded in zip(frames,values):
                frame[[f'{col}_TE100' for col in keys[0]]] = encoded.astype('float32')
                assert np.isfinite(frame.to_numpy(dtype=float)).all()
            return frames
        ''')
        pair(cells,'固定历史训练参数','学习率0.005需要更多树。subsample_freq=0表示历史配置没有启用按轮行采样，即使subsample写了0.8，也不能擅自改为1。列采样使列顺序影响复现。', 'model_params = '+repr(rparams)+'\npatience, expected_count = 700, 159')
    else:
        # 将大历史单元按顶层函数拆开，便于逐段学习。
        parts=[hybrid_features[:hybrid_features.index('def make_unsupervised_base')],hybrid_features[hybrid_features.index('def make_unsupervised_base'):hybrid_features.index('def prepare_hybrid_fold')],hybrid_features[hybrid_features.index('def prepare_hybrid_fold'):]]
        explanations=['先在原始EV数据上按单个特征计算购买均值。这里不读取比赛验证标签。两种最终模型对原始数据清洗规则不同，按历史实现保留。','将收入取整，通勤乘10再四舍五入；用余数提取个位、十位及尾数，用整除形成多尺度分组。阈值标志来自公开配方；不要根据本轮验证标签重新修改边界。返回特征表、编码key、收入和通勤数组。','先构造三份原始特征；频率和类别词表只从训练折学习。auto、10、100三种平滑分别编码同一组key。zip把三份特征与三份编码一一对应；astype float32保持历史精度和内存规模。']
        for i,(part,exp) in enumerate(zip(parts,explanations)):
            pair(cells,['准备原始数据均值','构造数字、分组和阈值特征','折内频率、类别与三组目标编码'][i],exp,part)
        pair(cells,'固定历史参数','深度5、32叶、学习率0.02、三组编码。四线程CPU；这里subsample_freq=1，和另一个模型不同，必须保留。',hparams_code+'\nprepare_fold = prepare_hybrid_fold\npatience, expected_count = 500, 110')
    urls={'baseline':['https://lightgbm.readthedocs.io/en/stable/Parameters.html'],'neighborhood':['https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94587-lb-0-94612?scriptVersionId=346904855','https://www.kaggle.com/code/jazivxt/single-model-zoom-zoom','https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637'],'hybrid':['https://www.kaggle.com/code/megayak/s6e9-one-lightgbm-from-raw-data-cv-0-9463']}[kind]
    pair(cells,'五折训练并回填预测','这是主要耗时单元。每折留出五分之一用于评价；早停也使用这个验证集，因此OOF属于开发评价。predict_proba的[:,1]取购买概率。五次覆盖完成后每行coverage应为1。不要为了整理日志重训。',f'run_name = {kind!r}\nmethod_sources = {urls!r}\n'+TRAIN)
    pair(cells,'保存并交接真实结果','只保存完整OOF、测试预测和摘要。输出目录已经存在时停止，先保存上次结果后重开会话。把整个目录保存为Notebook输出，告诉融合负责人挂载。',EXPORT)
    cells.append(md('## 中文结果解析与理解检查\n\n逐折AUC比较必须使用相同fold。整体OOF AUC和五折AUC的平均不是同一个量。两个完整方案同时改变多组特征，不能宣称某一列造成全部提升。\n\n请回答：为什么测试集不参与早停？为什么目标编码训练行不能直接使用全训练折groupby均值？为什么相同特征数量仍可能有不同列序？请画出一行样本从原始数据到验证预测经过的步骤。\n\n运行后用实际输出写中文观察；摘要已自动生成英文Report Summary。历史参照：收入邻域模型0.946046626，混合特征模型0.946129123；未训练前不能把这些数写成本次结果。'))
    save({'baseline':'01_raw_feature_lightgbm_baseline.ipynb','neighborhood':'02_neighborhood_multiscale_lightgbm.ipynb','hybrid':'03_hybrid_target_encoding_lightgbm.ipynb'}[kind],cells)

cells=intro('排名融合与复现检查','C主实现，A/B独立核对','官方数据、共享Dataset、三位成员保存的baseline/neighborhood/hybrid输出目录；最后可额外挂载历史两模型输出作核对','本项目自训练预测组合。40/60是历史冻结部署比例，50/50是对照。已有OOF上的选权属于开发评估，基础OOF训练集存在交叉依赖，不能称严格端到端nested CV。')
guide(cells)
pair(cells,'读取数据及共享分组','先统一行序，再组合分数。',COMMON+'\n'+LOAD_FOLDS)
pair(cells,'定位三个新运行并严格对齐','通过summary里的描述性run_name选择输出。历史文件没有这些新名称，不会替代新训练。重复挂载多个运行时明确停止，由成员选择唯一版本。', '''
def read_new_run(name):
    directories=[]
    for p in INPUT.rglob('run_summary.json'):
        metadata=json.loads(p.read_text())
        if metadata.get('run_name')==name:
            directories.append(p.parent)
    assert len(directories)==1, (name,directories)
    directory=directories[0]
    oof=align_rows(pd.read_csv(directory/'oof_predictions.csv'),train.id)
    sub=align_rows(pd.read_csv(directory/'submission.csv'),test.id)
    assert np.array_equal(oof.target,y) and np.array_equal(oof.fold,fold_ids)
    assert oof.prediction.between(0,1).all() and sub[TARGET].between(0,1).all()
    return oof.prediction.to_numpy(),sub[TARGET].to_numpy(),str(directory)
runs={name:read_new_run(name) for name in ['baseline','neighborhood','hybrid']}
''')
pair(cells,'理解全局排名','[0.2,0.8,0.8]的平均并列名次是[1,2.5,2.5]。除以3得到百分位排名。这里使用整个OOF的排名，不能groupby(fold)后排名。排名融合分数并非校准后的购买概率。', '''
from scipy.stats import rankdata
def percentile_rank(values):
    """输入有限的一维预测；输出保留并列关系的全局百分位排名。"""
    values=np.asarray(values,dtype=float)
    assert values.ndim==1 and len(values)>0 and np.isfinite(values).all()
    return rankdata(values,method='average')/len(values)
assert np.allclose(percentile_rank([0.2,0.8,0.8]),[1/3,2.5/3,2.5/3])
left,right=percentile_rank(runs['neighborhood'][0]),percentile_rank(runs['hybrid'][0])
print('Rank correlation:',np.corrcoef(left,right)[0,1])
''')
pair(cells,'复现选权并计算固定融合','先计算全局排名，再切四折开发集选择混合特征模型的权重；41个候选间隔0.025。AUC相同时选择靠近0.5的权重，仍相同则选择较小权重以保证确定性。每折留出评价仍属于已有OOF上的开发检查。最终提交保持历史40/60，新中位数另行记录。', '''
selection=[]
selected_oof=np.empty(len(train))
for fold in range(5):
    development=fold_ids!=fold
    held_out=~development
    scores=[(float(w),float(roc_auc_score(y[development],(1-w)*left[development]+w*right[development]))) for w in np.linspace(0,1,41)]
    weight,score=min(scores,key=lambda item:(-item[1],abs(item[0]-0.5),item[0]))
    selected_oof[held_out]=(1-weight)*left[held_out]+weight*right[held_out]
    selection.append({'held_out_fold':fold,'hybrid_weight':weight,'development_auc':score,
                      'held_out_auc':float(roc_auc_score(y[held_out],selected_oof[held_out]))})
median_weight=float(np.median([r['hybrid_weight'] for r in selection]))
candidates={name:values[0] for name,values in runs.items()}
candidates.update(equal_rank=0.5*left+0.5*right,frozen_rank=0.4*left+0.6*right,
                  selected_weights=selected_oof,median_rank=(1-median_weight)*left+median_weight*right)
overall={name:float(roc_auc_score(y,pred)) for name,pred in candidates.items()}
fold_results=[{'fold':fold,**{name:float(roc_auc_score(y[fold_ids==fold],pred[fold_ids==fold])) for name,pred in candidates.items()}} for fold in range(5)]
display(pd.DataFrame(selection))
display(pd.DataFrame(fold_results))
print(overall,'New median hybrid weight:',median_weight)
''')
pair(cells,'保存40/60提交与复现证据','两份输入submission已经各自平均五折概率，直接对各自测试预测排名后加权。保存一份最终提交，不自动上传。权重表放入summary，避免重复文件。', '''
output=Path('/kaggle/working/final_rank_blend')
output.mkdir(exist_ok=False)
final_test=0.4*percentile_rank(runs['neighborhood'][1])+0.6*percentile_rank(runs['hybrid'][1])
submission=sample.copy()
submission[TARGET]=final_test
assert submission.id.equals(test.id) and submission[TARGET].between(0,1).all()
submission.to_csv(output/'submission.csv',index=False)
pd.DataFrame({'id':train.id,'target':y,'fold':fold_ids,'prediction':candidates['frozen_rank']}).to_csv(output/'oof_predictions.csv',index=False)
report=(f'Two newly trained LightGBM models were combined using global percentile ranks and historical frozen weights 0.40 and 0.60. '
        f'The frozen blend OOF ROC AUC was {overall["frozen_rank"]:.9f}; the equal-weight control was {overall["equal_rank"]:.9f}. '
        'Weight selection over existing OOF predictions is a development analysis, not end-to-end nested cross-validation. '
        'Public and private performance for the new submission remains unverified.')
summary={'weights':{'neighborhood':0.4,'hybrid':0.6},'new_median_hybrid_weight':median_weight,
         'input_directories':{k:v[2] for k,v in runs.items()},'oof_metrics':overall,'fold_metrics':fold_results,
         'weight_selection':selection,'versions':current_versions(),'public_score':None,'report_summary':report}
(output/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
print(report)
print(output)
''')
pair(cells,'核对两套历史预测','最终提交前挂载数据说明中列出的两个历史输出。按ID比较最大绝对差、平均绝对差及AUC，同时对齐特征列和环境。历史文件缺失则明确报错，不能声称完成逐行一致性验收。原最终提交文件不在本包内；下面重建历史40/60文件作数值对照。', '''
checks=[]
historical_test=[]
for name in ['neighborhood','hybrid']:
    directory=Path(foundation_note['history_directories'][name])
    assert directory.is_dir(), f'Attach historical output: {directory}'
    old=align_rows(pd.read_csv(directory/'oof_predictions.csv'),train.id)
    old_sub=align_rows(pd.read_csv(directory/'submission.csv'),test.id)
    assert np.array_equal(old.target,y) and np.array_equal(old.fold,fold_ids)
    column='probability' if 'probability' in old else 'prediction'
    difference=np.abs(runs[name][0]-old[column].to_numpy())
    test_difference=np.abs(runs[name][1]-old_sub[TARGET].to_numpy())
    historical_test.append(old_sub[TARGET].to_numpy())
    checks.append({'model':name,'old_auc':float(roc_auc_score(y,old[column])),
                   'new_auc':overall[name],'max_oof_difference':float(difference.max()),
                   'mean_oof_difference':float(difference.mean()),'max_test_difference':float(test_difference.max())})
old_blend=0.4*percentile_rank(historical_test[0])+0.6*percentile_rank(historical_test[1])
summary['reproduction_checks']=checks
summary['max_final_difference_vs_reconstructed_history']=float(np.max(np.abs(final_test-old_blend)))
(output/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
display(pd.DataFrame(checks))
print('Final max difference:',summary['max_final_difference_vs_reconstructed_history'])
''')
cells.append(md('## 分析与交接\n\n历史参照：两单模OOF为0.946046626、0.946129123，50/50为0.946192671，40/60为0.946196333，历史公开榜0.94642。实际不一致先查输入版本、fold、列顺序、数据类型、编码和参数。接近或相同OOF不保证测试预测相同。\n\nA审核后手动提交保存的submission，并把真实成绩更新到summary；私人榜未公布就写未知。三人分别解释：B说明编码标签范围，A说明邻域统计，C说明排名和选权。\n\n来源：[排名文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)、[嵌套验证说明](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。历史Notebook22的诊断使用(rank-0.5)/n，交接文档的最终文件使用rank/n；固定同一权重和样本数时二者相差共同常数，AUC不变，但文件数值不同。本文件遵循最终交接文档rank/n。'))
save('04_rank_blend_and_reproduction_checks.ipynb',cells)
print('Built five notebooks in',DEST)
