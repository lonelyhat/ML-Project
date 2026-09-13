"""Static source parity and small numerical checks; does not train Kaggle models."""
import ast
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def read_code(path):
    nb=json.loads(path.read_text(encoding='utf-8'))
    return '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')

for path in sorted(HERE.glob('*.ipynb')):
    nb=json.loads(path.read_text(encoding='utf-8'))
    assert nb['nbformat']==4
    assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
    for c in nb['cells']:
        if c['cell_type']=='code':
            ast.parse(c['source'])
            assert c['execution_count'] is None and c['outputs']==[]
    print(path.name, 'syntax and empty-output checks passed')

def function(text,name):
    return next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)

old_hybrid=read_code(ROOT/'notebooks/22-hyl001-hybrid-lightgbm.ipynb')
new_hybrid=read_code(HERE/'03_hybrid_target_encoding_lightgbm.ipynb')
for old,new in [('make_unsupervised_base','make_unsupervised_base'),('prepare_fold','prepare_hybrid_fold')]:
    a,b=function(old_hybrid,old),function(new_hybrid,new)
    b.name=a.name
    assert ast.dump(a)==ast.dump(b), old
print('Hybrid feature functions: exact AST parity with historical source')

old_helper=(ROOT/'notebooks/r002_features.py').read_text(encoding='utf-8')
new_helper=read_code(HERE/'02_neighborhood_multiscale_lightgbm.ipynb')
for old,new in [('build_r001_features','build_public_features'),('prepare_r001_fold','prepare_public_fold'),('income_neighborhood_table','income_neighborhood_table'),('prepare_income_neighborhoods','prepare_income_neighborhoods')]:
    a,b=function(old_helper,old),function(new_helper,new)
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='build_public_features': node.id='build_r001_features'
            return node
    b=Rename().visit(b)
    b.name=a.name
    assert ast.dump(a)==ast.dump(b), old
print('Neighborhood/base feature functions: exact AST parity after descriptive renaming')

table=function(new_helper,'income_neighborhood_table')
scope={'np':np}
exec(compile(ast.Module(body=[table],type_ignores=[]),'<table-test>','exec'),scope)
actual=scope['income_neighborhood_table'](np.array([0,0,1,2]),np.array([0.,1.,1.,0.]),3)
assert actual.shape==(3,7) and np.isfinite(actual).all()
assert np.allclose(actual[:,0],[(1+5)/12,(1+5)/11,5/11])
assert np.allclose(actual[:,4],actual[:,2]-actual[:,1])
print('Neighborhood count, smoothing, boundaries and slope: small numerical checks passed')

# 对齐函数用真实pandas测试：打乱恢复、缺失ID和重复ID拒绝。
try:
    import pandas as pd
    scope={'pd':pd,'np':np}
    align=function(read_code(HERE/'00_data_and_shared_validation.ipynb'),'align_rows')
    exec(compile(ast.Module(body=[align],type_ignores=[]),'<alignment-test>','exec'),scope)
    ids=pd.Series([10,20,30])
    result=scope['align_rows'](pd.DataFrame({'id':[30,10,20],'p':[.3,.1,.2]}),ids)
    assert np.allclose(result.p,[.1,.2,.3])
    for invalid in [pd.DataFrame({'id':[10,10,30]}),pd.DataFrame({'id':[10,20]})]:
        try: scope['align_rows'](invalid,ids)
        except AssertionError: pass
        else: raise AssertionError('Invalid IDs were accepted')
    print('Real pandas ID alignment and invalid-input checks passed')
except ImportError:
    print('Pandas checks unavailable in this interpreter')
print('Full Kaggle execution, LightGBM training and leaderboard reproduction remain untested.')
