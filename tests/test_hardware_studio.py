import pandas as pd
from ai_accel_lab.hardware import detect_hardware,recommended_backends
from ai_accel_lab.studio.results import ResultStore


def test_hardware_report_has_architecture():
    h=detect_hardware()
    d=h.to_dict()
    assert 'architecture' in d
    assert isinstance(recommended_backends(h),list)


def test_result_store(tmp_path):
    s=ResultStore(tmp_path)
    rid,csv,js=s.save(pd.DataFrame([{'a':1}]),'unit')
    assert csv.exists() and js.exists()
    assert len(s.list_runs())==1
