import csv
from pathlib import Path
p=Path('LOG00295.TXT')
first_bad={1:None,2:None}
with p.open('r',encoding='utf-8') as f:
    reader=csv.reader(f)
    for i,r in enumerate(reader, start=1):
        for col in [1,2]:
            if first_bad[col] is None:
                if len(r)<=col:
                    first_bad[col]=(i,'missing',r)
                else:
                    try:
                        float(r[col])
                    except Exception:
                        first_bad[col]=(i,r[col],r)
        if all(v is not None for v in first_bad.values()):
            break
print('first bad:',first_bad)
