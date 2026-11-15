import pytest
from audit_conflicts import sizes, dimension_status, code_status, labelled_codes, audit
from matching import identifiers
import pandas as pd


@pytest.mark.parametrize("a,b,status", [
    ("fan 120mm","fan 140mm","different_value_unverified_role"),
    ("drum 14 Inch","drum 13 in.","different_value_unverified_role"),
    ("fan 120mm","fan","missing_one"),
    ("model A1234","model B2345","missing_both"),
    ("14cm","140mm","different_units"),
    ("27,0 inch","27.0 in.","same_value_unverified_role"),
    ("24-70mm","50mm","ambiguous"),
    ("24 – 70mm","50mm","ambiguous"),
    ("16:9in","27in","ambiguous"),
    ("20 x 30mm","30mm","ambiguous"),
    ("20mm x 30mm","30mm","ambiguous"),
    ("lens 24mm and filter 67mm","lens 24mm","ambiguous"),
    ("Dymo Black - 9mm","Dymo 12mm","different_value_unverified_role"),
    ("Nikon FX 50mm","Nikon 50mm","same_value_unverified_role"),
    ("100mm lens","3.5mm headset","different_value_unverified_role"),
    ("123 456","123 789","missing_both"),
])
def test_size_audit(a,b,status):
    assert dimension_status(a,b)==status


def test_missing_not_disagreement():
    assert code_status("H390","H570e")=="missing_one"
    assert code_status("123","456")=="missing_both"


def test_known_identifier_limits():
    assert not identifiers("H390 T2U T9UH")
    assert identifiers("ML120") != identifiers("ML140")
    assert identifiers("DS-7204HQHI-K1")==identifiers("DS7204HQHIK1")
    assert not identifiers("DS 7204 HQHI K1")
    assert not identifiers("DDR4 1000KV 3.5mm")
    assert identifiers("AC1900 1080p 27-INCH")=={"ac1900","1080p","27inch"}


def test_model_marker_is_not_authority():
    assert labelled_codes("Model: AB-1234")==["ab-1234"]
    assert labelled_codes("MPN=AB1234")==["ab1234"]
    assert not labelled_codes("great model performance 1234")


def test_disjoint_codes_not_verified_conflict():
    p=dict(pair_id="a#b",outer_fold=0,label=1,is_hard_negative=False,
           title_left="Model: AB1234 120mm",title_right="Model: CD5678 140mm",
           sim_identifier=0.,sim_title=.8,fixed_prediction=1,renorm_prediction=1,
           renorm_score=.5,renorm_threshold=.13)
    result=audit(pd.DataFrame([p]))
    assert result.iloc[0].identifier_status=="disjoint_singletons"
    assert not result.iloc[0].verified_conflict
    assert result.iloc[0].renorm_score==.5
    assert result.iloc[0].renorm_prediction==1
    assert result.iloc[0].renorm_tp


def test_non_scalar_and_missing():
    assert sizes(None)=={"values":[],"ambiguous":False}
    assert sizes("120mm 120mm")["ambiguous"] is False
