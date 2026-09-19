from __future__ import annotations
import json,time,uuid
from pathlib import Path
import pandas as pd

class ResultStore:
    def __init__(self,root='results'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def save(self,df:pd.DataFrame,kind='benchmark',metadata=None):
        rid=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        csv=self.root/f'{kind}-{rid}.csv'; js=self.root/f'{kind}-{rid}.json'
        df.to_csv(csv,index=False)
        js.write_text(json.dumps({'id':rid,'kind':kind,'metadata':metadata or {},'records':df.to_dict('records')},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        return rid,csv,js
    def list_runs(self):
        return sorted(self.root.glob('*.json'),reverse=True)
