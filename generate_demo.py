import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
dates = pd.bdate_range("2025-01-01", periods=300)
rows=[]
for symbol, name, industry, drift in [
    ("600584","示例科技A","半导体",0.0008),
    ("000938","示例科技B","计算机",0.0005),
    ("600901","示例金融A","非银金融",0.0003),
    ("002945","示例券商A","证券",0.0002),
]:
    p=20+np.random.rand()*20
    for i,d in enumerate(dates):
        r=drift+np.random.normal(0,0.018)
        o=p*(1+np.random.normal(0,0.004))
        c=max(1,o*(1+r))
        h=max(o,c)*(1+abs(np.random.normal(0,0.008)))
        l=min(o,c)*(1-abs(np.random.normal(0,0.008)))
        v=max(100000, int(np.random.lognormal(14,0.5)))
        rows.append({"date":d.strftime("%Y-%m-%d"),"symbol":symbol,"name":name,"industry":industry,"open":o,"high":h,"low":l,"close":c,"volume":v,"amount":v*c,"turnover_rate":np.random.uniform(1,12)})
        p=c
out=Path(__file__).with_name("demo_market.csv")
pd.DataFrame(rows).to_csv(out,index=False,encoding="utf-8-sig")
print(out)
