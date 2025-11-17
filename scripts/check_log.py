import csv
import math
from pathlib import Path
p=Path('LOG00295.TXT')
rows=[]
with p.open('r',encoding='utf-8') as f:
    reader=csv.reader(f)
    for i,r in enumerate(reader):
        if not r: continue
        try:
            t=float(r[0])
        except Exception:
            continue
        if 77000<=t<=79000 or 97000<=t<=99000:
            rows.append((i+1,t,r))

print('Found',len(rows),'rows in target ranges')
for ln,t,r in rows[:50]:
    def check(val):
        try:
            float(val)
            return True
        except:
            return False
    ok1=check(r[1]) if len(r)>1 else False
    ok2=check(r[2]) if len(r)>2 else False
    print(f'line {ln}: t={t}, col1_ok={ok1}, col2_ok={ok2}, cols={len(r)}')
    print('  sample:', r[:8])
