import os
import glob

SRC = "/mnt/newdisk/anass/monthly_data"
DST = "/mnt/newdisk/anass/monthly_data_tsfix"
os.makedirs(DST, exist_ok=True)

def fix_year(ts: str) -> str:
    ts = (ts or "").strip()
    if len(ts) >= 4 and ts[:4].isdigit():
        return "2020" + ts[4:]   # year reinterpretation
    return ts

for fp in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
    out = os.path.join(DST, os.path.basename(fp))
    with open(fp, "r", encoding="utf-8", errors="ignore") as fin, open(out, "w", encoding="utf-8") as fout:
        header = fin.readline()
        fout.write(header)
        cols = header.strip().split(",")
        try:
            t_idx = cols.index("error_time")
        except ValueError:
            raise RuntimeError(f"'error_time' column not found in {fp}")

        for line in fin:
            parts = line.rstrip("\n").split(",")
            if len(parts) > t_idx:
                parts[t_idx] = fix_year(parts[t_idx])
            fout.write(",".join(parts) + "\n")

print("Done. Fixed files written to:", DST)

