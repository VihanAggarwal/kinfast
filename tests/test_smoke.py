# tests/test_smoke.py
import kinfast

def test_version_present():
    assert isinstance(kinfast.__version__, str)
    assert kinfast.__version__.count(".") >= 1
