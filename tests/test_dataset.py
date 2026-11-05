from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cad_inspector.dataset import load_case

def test_case2_loads():
    case=load_case(Path(__file__).resolve().parents[1]/"data/sample/AI_sample_CAD/TestCase2")
    assert len(case["parts"])>0
    assert case["real"].ndim==3
