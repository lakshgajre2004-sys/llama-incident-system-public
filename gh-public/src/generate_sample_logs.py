import pandas as pd
import os, random
from datetime import datetime, timedelta

os.makedirs('../sample_logs', exist_ok=True)

df = pd.read_csv('../datasets/sample_dataset_1000.csv')

base_time = datetime.now() - timedelta(days=7)
levels = ["INFO","WARN","ERROR","DEBUG"]
components = ["auth","db","api","worker","scheduler"]

def make_log_line(idx, row):
    t = base_time + timedelta(seconds=idx*10 + random.randint(0,9))
    ts = t.strftime("%Y-%m-%d %H:%M:%S")
    lvl = random.choices(levels, weights=[0.6,0.2,0.1,0.1])[0]
    comp = random.choice(components)
    msg = f"metricCpu={row.get('f0',0):.3f} metricMem={row.get('f1',0):.3f} target={int(row.get('target',0))}"
    return f"{ts} [{lvl}] {comp} - {msg}"

lines = []
for i, row in df.iterrows():
    lines.append(make_log_line(i, row))

with open('../sample_logs/sample_aggregated.log','w',encoding='utf-8') as f:
    f.write("\n".join(lines))

for comp in components:
    comp_lines = [ln for ln in lines if f"] {comp} -" in ln]
    if comp_lines:
        with open(os.path.join('../sample_logs', f'sample_{comp}.log'),'w',encoding='utf-8') as f:
            f.write("\n".join(comp_lines))

print('Sample logs created in ../sample_logs/')
