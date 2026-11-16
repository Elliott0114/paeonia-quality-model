"""Portable bindings for the unmodified regional model and trainer."""
import os,sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import base as b
from backbones import build_1d_backbone_model
old=SimpleNamespace(build_1d_backbone_model=build_1d_backbone_model)
CORE=b.CORE
OUT=Path(os.environ.get('PAEONIA_OUTPUT','outputs/cm-v4-cnn')).resolve()
dump=b.dump;digest=b.w.digest;read=b.w.read_csv;write=b.w.write_csv;now=b.now
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
