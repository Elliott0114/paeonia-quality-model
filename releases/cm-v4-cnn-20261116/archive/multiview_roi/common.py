"""Isolated helpers; old sources are imported read-only."""
import os,sys,json,hashlib,importlib.util
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PROJECT=REPO.parents[1]
AUDIT=PROJECT/'04_核查记录'
OUT=AUDIT/HERE.name
sys.path.insert(0,str(REPO/'experiments/model_search_v4_grouped_20261028'))
import shared as old
sys.path.insert(0,str(HERE))
import numpy as np
import pandas as pd
import torch
b=old.b;CORE=old.CORE
SOURCE=AUDIT/'g_adjustment_full_v4_20261031'
EFFICIENT=AUDIT/'efficientnet_v4_grouped_20261026'
RLIB='/Users/elliottlv/Library/R/paeonia-roi-jive/4.6/library'
ALPHAS=[.01,.1,1.,10.,100.,1000.]
MODELS=['ridge_p0','ridge_dispersion','efficientnet_p0','mil_mean','mil_attention']
NEW=['mil_mean','mil_attention']
dump=b.dump;digest=b.w.digest;read=b.w.read_csv;write=b.w.write_csv;now=b.now

def load():
    t,x,y,ss=b.load_inputs(OUT)
    return t,x,y,ss

def spatial():
    with np.load(OUT/'inputs/regions.npz',allow_pickle=False) as z:
        return z['spectra'].astype(float),z['weights'].astype(float)

def dispersion():
    with np.load(OUT/'inputs/dispersion.npz',allow_pickle=False) as z:return z['features'].astype(float)

def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def import_file(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def source_paths():
    m=json.loads((AUDIT/'analysis_configured_baseline_20261022/run.json').read_text())
    archive=Path(m['source_paths']['cohort']).parents[1]
    hreb=archive/'02_FULL_REPRODUCIBILITY_PACKAGE/PROCESSED_INPUTS/hsi_target_blind_rebuild/data'
    c01=PROJECT/'02_数据与分析/液相-高光谱分析/baishao_hsi_lc_project/outputs/c01_triplet_preprocessed'
    cube=PROJECT/'02_数据与分析/液相-高光谱分析/data/高光谱数据/浙江中医药邵逸夫白芍数据采集/中药饮片反射率数据'
    return dict(hreb=hreb,c01=c01,cube=cube,visible=Path(m['source_paths']['visible15']),
        photo=AUDIT/'analysis_configured_baseline_20261022/visible/outer_predictions.csv.gz',
        extractor=archive/'03_SOURCE_DATA_AND_CODE/code/rebuild_target_blind_hsi_qc.py')

def check_sources():
    m=json.loads((OUT/'manifest.json').read_text())
    for k in ['sources','protected']:
        for p,h in m[k].items():assert digest(Path(p))==h,p
    for n,h in m['inputs'].items():assert digest(OUT/'inputs'/n)==h,n
    for p,s in m['large_source_metadata'].items():
        st=Path(p).stat();assert [st.st_size,st.st_mtime_ns]==s,p
    return m

def safe_json(df):return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=15))
