import os

import numpy as np
import pandas as pd

from utils.time_utils import get_current_time

output_dir = os.path.join(os.path.dirname(__file__), "output")
files = os.listdir(output_dir)

dfs = []

for file in files:
    if not file.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(output_dir, file))
    dfs.append(df)

ensemble = pd.concat(dfs, axis=1)
row_mean = ensemble.mean(axis=1)
row_mean.to_csv(os.path.join(output_dir, f"ensemble_{get_current_time()}.csv"), index=False)

