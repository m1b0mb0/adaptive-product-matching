import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import final_evaluation as e


@pytest.mark.parametrize("y,p",[
 ([0,0,1,1],[0,1,0,1]),([0,1],[0,1]),([0,1],[1,0]),
 ([0,0],[0,0]),([1,1],[0,0]),([1,1],[1,1]),([],[])])
def test_metrics(y,p):
    s=e.measures(y,p)
    assert s["confusion_matrix"]==(confusion_matrix(y,p,labels=[0,1]).tolist() if y else [[0,0],[0,0]])
    if len(y):
        a,b,c,_=precision_recall_fscore_support(y,p,average="binary",zero_division=0)
        assert (s["precision"],s["recall"],s["f1"])==pytest.approx((a,b,c))


def test_hard_missing_and_transitions():
    s=e.measures([0,0,0,1],[1,0,1,1],[True,False,None,False],[True,False,False,False])
    assert s["hard_fpr"]==1 and s["other_fpr"]==0
    assert s["unknown_hard_flag_negatives"]==1 and s["no_evidence_positive"]==1
    assert e.transitions([0,0,1,1],[1,0,0,1],[0,1,1,0])==dict(
        disagreements=4,corrected_fp=1,added_fp=1,corrected_fn=1,added_fn=1)


@pytest.fixture(scope="module")
def frozen():
    return e.preflight()


def test_actual_preflight_and_no_fit(frozen):
    c,b,train,p=frozen
    assert p["status"]=="PASSED" and not p["test_opened"]
    with e.no_fit():
        x=b["extractor"].transform(train[:5]).matrix
        scores=e.score_all(x,b,c)
        for name in e.METHODS:
            assert len(scores[name])==5
        with pytest.raises(RuntimeError,match="forbidden"):
            b["extractor"].fit(train[:2])
    np.testing.assert_allclose(scores["fixed"],e.evidence_coverage(x)*scores["renorm"],atol=1e-14)


def test_empty_oov_zero_are_distinct(frozen):
    c,b,_,_=frozen
    with e.no_fit():
        x=b["extractor"].transform([{},{"title_left":"ЖЖЖЖ","title_right":"ЖЖЖЖ"},
            {"brand_left":"A","brand_right":"B"}]).matrix
        assert np.isnan(x[:2]).all()
        assert x[2,2]==0 and not np.isnan(x[2,2])
        values=e.score_all(x,b,c)
        for m in e.METHODS:
            if m!="logistic":assert np.all(values[m][:2]==0)
        assert values["adaptive"][2]==pytest.approx(.17)
        assert values["adaptive_corrected"][2]==0
        np.testing.assert_array_equal(values["logistic"],b["logistic"].predict_score(x))


def pair(a,b,ca,cb):
    return dict(pair_id=f"{a}#{b}",id_left=a,id_right=b,cluster_id_left=ca,
                cluster_id_right=cb,title_left="Alpha",title_right="Beta",
                label=int(ca==cb),is_hard_negative=False)


def test_integrity_and_leakage():
    train=[pair("a","b","1","1")]
    rows=[pair("c","d","2","3")]
    s,errors=e.validate_structure(rows,train);assert not errors
    assert e.validate_structure(rows+rows,train)[1]
    assert "offers_overlap" in e.validate_structure(train,train)[1]


def test_report_before_test(tmp_path):
    stats={m:e.measures([0,1],[0,1],[True,False]) for m in e.METHODS}
    records=[dict(scenario=s,missing_rate=r,seed=seed,method=m,**stats[m])
             for s in e.SCENARIOS for r in e.RATES for seed in e.SEEDS for m in e.METHODS]
    pd.DataFrame(records).to_csv(tmp_path/"missingness_results.csv",index=False)
    result=dict(metrics=stats,paired_comparisons=[
        dict(from_method="fixed",to_method="renorm",**e.transitions([0,1],[0,1],[0,1]))])
    e.report(tmp_path,result)
    assert (tmp_path/"test_results.md").exists()
    assert len(list(tmp_path.glob("*.png")))==3
    assert pd.read_csv(tmp_path/"missingness_summary.csv").f1_std.eq(0).all()


def test_stale_receipt_blocks_before_test(tmp_path,monkeypatch):
    monkeypatch.setattr(e,"OUT",tmp_path)
    e.save(tmp_path/"pretest_validation.json",{"status":"FAILED"})
    with pytest.raises(ValueError,match="receipt"):
        e.execute({},None,None,{})
    assert not (tmp_path/"execution_started.json").exists()
