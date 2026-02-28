import pandas as pd
import glob

files = sorted(glob.glob('data/*.csv'))
for f in files[-4:]:
    df = pd.read_csv(f)
    col = 'FTR' if 'FTR' in df.columns else ('Result3' if 'Result3' in df.columns else None)
    if col:
        print(f"\n{f}  (total: {len(df)})")
        print(df[col].value_counts().to_dict())
    else:
        print(f"\n{f} — ไม่พบ FTR หรือ Result3, columns:", list(df.columns)[:10])
