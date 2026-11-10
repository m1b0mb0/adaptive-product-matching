"""Finalization checks use software fixtures, never fabricated study results."""
import csv
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

import prepare_final_method as final
from audit_wdc import offers
from matching import FeatureExtractor


def fixture_pairs():
    records=[]
    for i in range(18):
        base={"title":f"Camera Model AX{i:03d} item", "brand":"Example", "description":f"Camera description {i}"}
        pair={"id_left":f"a{i}","id_right":f"b{i}","cluster_id_left":f"e{i}","cluster_id_right":f"e{i}","label":1,"is_hard_negative":False}
        for side in ("left","right"):
            pair.update({f"{key}_{side}":value for key,value in base.items()})
        pair["pair_id"]=pair["id_left"]+"#"+pair["id_right"]
        records.append(pair)
    for i in range(18):
        for j in range(i+1,18):
            pair={"id_left":f"a{i}","id_right":f"a{j}","cluster_id_left":f"e{i}","cluster_id_right":f"e{j}","label":0,"is_hard_negative":bool((i+j)%2)}
            for index,side in ((i,"left"),(j,"right")):
                pair.update({f"title_{side}":f"Camera Model AX{index:03d} item",f"description_{side}":f"Camera description {index}",f"brand_{side}":"Example"})
            pair["pair_id"]=pair["id_left"]+"#"+pair["id_right"]
            records.append(pair)
    return records


def test_inner_aggregation_has_one_equal_weight_row_per_pair():
    indices,y,values,counts=final.aggregate_inner_calibration([(4,1,.2),(2,0,.1),(4,1,.8)])
    assert indices==[2,4] and y.tolist()==[0,1] and counts==[1,2]
    np.testing.assert_allclose(values,[.1,.5])


def test_conflicting_inner_targets_cannot_be_averaged():
    with pytest.raises(ValueError,match="conflicting"):
        final.aggregate_inner_calibration([(1,0,.2),(1,1,.4)])


def test_predeclared_corrected_has_no_new_hyperparameter_search():
    protocol=final.protocol_spec()
    assert protocol["corrected_parameter_grid"]==[{"gamma":1.,"prior":0.,"penalty":0.}]
    assert protocol["threshold_grid"]==final.THRESHOLDS.tolist()
    assert "INNER" in protocol["final_calibration"]["source"]
    assert final._configuration_options(final.CORRECTED)==[{}]


def test_pipeline_fits_inside_folds_and_calibrates_only_inner_predictions(monkeypatch):
    pairs=fixture_pairs()
    plan=final.nested_plan(pairs)
    expected_fits=[]
    calibration_multiplicity=Counter()
    for outer in plan["outer_folds"]:
        outer_rows=[pairs[i]for i in outer["fit_indices"]]
        for inner in outer["inner_folds"]:
            train=[outer_rows[i]for i in inner["fit_indices"]]
            held=[outer_rows[i]for i in inner["eval_indices"]]
            assert not offers(train)&offers(held)
            expected_fits.append(frozenset(offers(train)))
            calibration_multiplicity.update(outer["fit_indices"][i]for i in inner["eval_indices"])
        expected_fits.append(frozenset(offers(outer_rows)))
    original_fit=FeatureExtractor.fit
    actual_fits=[]
    def spy(self,rows):
        actual_fits.append(frozenset(offers(rows)))
        return original_fit(self,rows)
    selections=[]
    original_select=final.select_from_oof
    def select_spy(y,options):
        selections.append(len(options))
        return original_select(y,options)
    monkeypatch.setattr(FeatureExtractor,"fit",spy)
    monkeypatch.setattr(final,"select_from_oof",select_spy)
    result,calibration=final.run_corrected_development(pairs,plan)
    assert actual_fits==expected_fits and len(actual_fits)==12
    assert selections[:3]==[1,1,1]
    assert len(selections)==3+len(final.ALL_METHODS)
    assert result["final_calibration"]["included_input_indices"]==sorted(calibration_multiplicity)
    for row in calibration:
        assert row["multiplicity"]==calibration_multiplicity[row["input_index"]]
    assert len(result["missingness"])==30
    for row in result["missingness"]:
        if row["missing_rate"]==0:
            assert row["confusion_matrix"]==result["clean_metrics"]["confusion_matrix"]
    assert len(result["outer_predictions"])==len(plan["outer_evaluation_indices"])
    for fold in result["folds"]:
        expected_threshold=fold["selected"]["threshold"]
        assert all(row["threshold"]==expected_threshold for row in result["outer_predictions"]if row["outer_fold"]==fold["outer_fold"])


def fake_cache(tmp_path):
    code=tmp_path/"original.py"
    canonical=b"x = 1\n"
    code.write_bytes(canonical.replace(b"\n",b"\r\n"))
    train=tmp_path/"train.jsonl"
    train.write_bytes(b"fixture only")
    pairs=[{"pair_id":"A#B"},{"pair_id":"C#D"}]
    plan={"outer_evaluation_indices":[0,1]}
    stats=final._stats(np.array([1,0]),np.array([1,0]))
    cached={"provenance":{"train_sha256":final.digest(train),"code_sha256":{"original.py":hashlib.sha256(canonical).hexdigest()},"packages":{}},"splits":plan,"clean_metrics":{m:stats for m in final.METHODS}}
    directory=tmp_path/"cache"
    directory.mkdir()
    (directory/"development_results.json").write_text(json.dumps(cached),encoding="utf-8")
    (directory/"protocol.json").write_text("{}",encoding="utf-8")
    with (directory/"clean_predictions.csv").open("w",encoding="utf-8",newline="")as stream:
        writer=csv.DictWriter(stream,fieldnames=["method","pair_id","label","prediction"])
        writer.writeheader()
        writer.writerows({"method":m,"pair_id":pid,"label":y,"prediction":y}for m in final.METHODS for pid,y in [("A#B",1),("C#D",0)])
    hashes={str(p.as_posix()):final.digest(p)for p in directory.iterdir()}
    (directory/"pipeline_validation.json").write_text(json.dumps({"artifact_sha256":hashes}),encoding="utf-8")
    return pairs,plan,directory,train,code


def test_cache_reuse_checks_predictions_and_accepts_only_line_ending_changes(tmp_path,monkeypatch):
    pairs,plan,directory,train,code=fake_cache(tmp_path)
    monkeypatch.chdir(tmp_path)
    _,checks=final.verify_historical_cache(pairs,plan,directory,train)
    assert checks["baseline_experiments_rerun"]is False
    assert checks["baseline_metrics_recomputed_from_csv"]is True
    code.write_text("x = 2\n",encoding="utf-8")
    with pytest.raises(ValueError,match="beyond CRLF/LF"):
        final.verify_historical_cache(pairs,plan,directory,train)


def test_changed_input_or_old_results_fail_closed(tmp_path,monkeypatch):
    pairs,plan,directory,train,code=fake_cache(tmp_path)
    monkeypatch.chdir(tmp_path)
    train.write_bytes(b"changed")
    with pytest.raises(ValueError,match="TRAIN does not match"):
        final.verify_historical_cache(pairs,plan,directory,train)


def test_main_rejects_named_nontrain_before_opening(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Non-TRAIN must not be read")
    monkeypatch.setattr(final,"digest",forbidden)
    with pytest.raises(SystemExit)as error:
        final.main(["--train","never-open-reserved.json.gz"])
    assert error.value.code==2
