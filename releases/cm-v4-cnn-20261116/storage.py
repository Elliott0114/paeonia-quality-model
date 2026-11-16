from pathlib import Path
import hashlib
import pandas as pd
import os

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require_writable_run(p):
    root=Path(os.environ["PAEONIA_OUTPUT"]).resolve()
    if not Path(p).resolve().is_relative_to(root):raise ValueError("Write outside the requested output directory")

def write_csv(frame: pd.DataFrame, path: Path) -> None:
    require_writable_run(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    frame.to_csv(tmp, index=False, compression='gzip' if path.suffix == '.gz' else None)
    tmp.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')
