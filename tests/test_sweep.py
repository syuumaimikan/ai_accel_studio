import pandas as pd
from ai_accel_lab.sweep import pareto_front

def test_pareto():
    df=pd.DataFrame({'x':[1,2,3],'y':[3,2,1]}); o=pareto_front(df,[('x','min'),('y','min')]); assert o.pareto.all()
