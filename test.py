import pandas as pd

df = pd.read_parquet('data/vlm_captions_redcaps_00.parquet')

print(df["vlm_caption"].head(1).values)