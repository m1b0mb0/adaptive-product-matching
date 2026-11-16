"""Frozen six-method evaluation. Default is preflight; --execute consumes TEST once."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from adaptive_corrected import METHOD_SPECIFICATION, corrected_scores
from audit_wdc import read_pairs, summary, offers, _canonical_pair, _signature
from logistic_baseline import SimilarityLogisticRegression
from matching import FeatureExtractor, Features, evidence_scores, evidence_coverage
from run_known_methods import INTEGRITY_FIELDS, apply_missingness, _representation_metadata
from prepare_final_method import json_digest
from study_method_selection import table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"docs/final_evaluation"
CONFIG=ROOT/"docs/final_method/frozen_configuration.json"
METHODS=("text","fixed","renorm","adaptive","logistic","adaptive_corrected")
TEST_NAME="wdcproducts80cc20rnd100un_gs.json.gz"
TEST_SHA="258f9b408e715410f07480bf4ad39788a10d5cee1512e322091de43dafae1297"
RATES=(0.,.1,.2,.3,.5)
SEEDS=(11,22,33)
SCENARIOS=("auxiliary","all_groups")


def digest(path,lf=False):
    data=Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n",b"\n") if lf else data).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def no_fit():
    stack=ExitStack()
    def reject(*args,**kwargs):
        raise RuntimeError("Fit is forbidden during frozen evaluation")
    for cls,name in [(FeatureExtractor,"fit"),(SimilarityLogisticRegression,"fit"),
                     (TfidfVectorizer,"fit"),(TfidfVectorizer,"fit_transform"),(LogisticRegression,"fit")]:
        stack.enter_context(patch.object(cls,name,reject))
    return stack


def measures(y,pred,hard=None,missing=None):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isin(pred,[0,1]).all():
        raise ValueError("Aligned binary vectors required")
    tp=int(((y==1)&(pred==1)).sum());fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum());tn=int(((y==0)&(pred==0)).sum())
    out=dict(n=len(y),positives=int(sum(y)),negatives=int(sum(y==0)),
             tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,
             recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
             confusion_matrix=[[tn,fp],[fn,tp]])
    for label,value in (("hard",True),("other",False)):
        mask=np.array([v is value for v in hard],bool)&(y==0) if hard is not None else np.zeros(len(y),bool)
        n=int(mask.sum());errors=int(pred[mask].sum())
        out.update({label+"_n":n,label+"_fp":errors,label+"_tn":n-errors,
                    label+"_fpr":errors/n if n else None})
    out["unknown_hard_flag_negatives"]=sum(v is not True and v is not False and a==0 for v,a in zip(hard,y)) if hard is not None else int(sum(y==0))
    missing=np.zeros(len(y),bool) if missing is None else np.asarray(missing,bool)
    out.update(no_evidence_n=int(missing.sum()),no_evidence_positive=int(pred[missing].sum()))
    return out


def transitions(y,a,b):
    y=np.asarray(y);a=np.asarray(a);b=np.asarray(b)
    return dict(disagreements=int(sum(a!=b)),
        corrected_fp=int(sum((y==0)&(a==1)&(b==0))),
        added_fp=int(sum((y==0)&(a==0)&(b==1))),
        corrected_fn=int(sum((y==1)&(a==0)&(b==1))),
        added_fn=int(sum((y==1)&(a==1)&(b==0))))


def score_all(x,bundle,config):
    selected=config["final_models"];info=bundle["informativeness"]
    values={m:evidence_scores(x,m) for m in ("text","fixed","renorm")}
    a=selected["adaptive"]
    values["adaptive"]=evidence_scores(x,"adaptive",info,gamma=a["gamma"],prior=a["prior"],penalty=a["penalty"])
    values["logistic"]=bundle["logistic"].predict_score(x)
    values["adaptive_corrected"]=corrected_scores(x,info)
    for m,s in values.items():
        if s.shape!=(len(x),) or not np.isfinite(s).all() or np.any((s<0)|(s>1)):
            raise ValueError("Invalid score "+m)
    return values


def validate_structure(rows,train=None):
    s=summary(rows)
    failures=[key for key in INTEGRITY_FIELDS if s[key]]
    if s["schema_keysets"]!=1 or s["cluster_labeled_pairs"]!=len(rows) or s["explicit_pair_ids"]!=len(rows):
        failures.append("incomplete or inconsistent schema")
    if train is not None:
        sets={
          "offers":lambda rs:offers(rs),
          "entities":lambda rs:{str(p["cluster_id_"+side]) for p in rs for side in ("left","right")},
          "pairs":lambda rs:{_canonical_pair(p) for p in rs},
          "text_signatures":lambda rs:{v for p in rs for side in ("left","right") if (v:=_signature(p,side,("title","description","brand")))},
          "title_signatures":lambda rs:{v for p in rs for side in ("left","right") if (v:=_signature(p,side,("title",)))}
        }
        s["train_test_overlaps"]={name:len(fn(rows)&fn(train)) for name,fn in sets.items()}
        failures += [name+"_overlap" for name in ("offers","entities","pairs") if s["train_test_overlaps"][name]]
    return s,failures


def preflight():
    c=json.loads(CONFIG.read_text("utf-8"))
    validation=json.loads((ROOT/"docs/final_method/pipeline_validation.json").read_text("utf-8"))
    if digest(CONFIG)!=validation["artifact_sha256"]["docs/final_method/frozen_configuration.json"]:
        raise ValueError("Frozen configuration changed")
    for name in ("method_specification.md","final_evaluation_protocol.md","implementation_status.md"):
        rel="docs/final_method/"+name
        if digest(ROOT/rel)!=validation["artifact_sha256"][rel]:
            raise ValueError("Frozen document changed: "+name)
    p=c["provenance"]
    if platform.python_version()!=p["python"]:raise ValueError("Python mismatch")
    for name,version in p["packages"].items():
        if importlib.metadata.version(name)!=version:raise ValueError("Package mismatch: "+name)
    for name,h in p["code_sha256_lf"].items():
        if digest(ROOT/name,True)!=h:raise ValueError("Code mismatch: "+name)
    if json_digest(c["final_models"])!=c["parameter_sha256"]:raise ValueError("Parameter hash mismatch")
    if c["method_specification"]!=METHOD_SPECIFICATION or json_digest(METHOD_SPECIFICATION)!=c["method_specification_sha256"]:
        raise ValueError("Specification mismatch")
    if set(c["final_models"])!=set(METHODS):raise ValueError("Six methods required")
    train_path=ROOT/p["train_path"]
    if digest(train_path)!=p["train_sha256"]:raise ValueError("TRAIN mismatch")
    artifact=c["artifacts"]["model_bundle"]
    if digest(ROOT/artifact["path"])!=artifact["sha256"]:raise ValueError("Bundle mismatch")
    bundle=joblib.load(ROOT/artifact["path"])
    if bundle["schema_version"]!=1 or bundle["selected"]!=c["final_models"] or bundle["train_sha256"]!=p["train_sha256"] or bundle["method_specification"]!=METHOD_SPECIFICATION:
        raise ValueError("Bundle metadata mismatch")
    np.testing.assert_array_equal(bundle["informativeness"],c["full_train_fit"]["informativeness"])
    if bundle["logistic"].metadata()!=c["full_train_fit"]["logistic"]:raise ValueError("LR metadata mismatch")
    train=read_pairs(train_path)
    stats,errors=validate_structure(train)
    if errors:raise ValueError("TRAIN integrity: "+str(errors))
    if _representation_metadata(bundle["extractor"],train)!=c["full_train_fit"]["representation"]:
        raise ValueError("Frozen representation mismatch")
    with no_fit():
        x=bundle["extractor"].transform(train[:12]).matrix
        score_all(x,bundle,c)
        missing=bundle["extractor"].transform([{},{"title_left":"ЖЖЖЖ","title_right":"ЖЖЖЖ"}]).matrix
        if not np.isnan(missing).all():raise ValueError("Missing/OOV smoke mismatch")
        scores=score_all(missing,bundle,c)
        if any(np.any(scores[m]) for m in METHODS if m!="logistic"):raise ValueError("No evidence handling")
    return c,bundle,train,dict(status="PASSED",test_opened=False,python=platform.python_version(),
         packages=p["packages"],train_sha256=p["train_sha256"],bundle_sha256=artifact["sha256"],
         config_sha256=digest(CONFIG),train_integrity=stats,
         code_sha256={name:digest(ROOT/name) for name in ("final_evaluation.py","tests/test_final_evaluation.py","study_method_selection.py")},
         frozen_code_sha256_lf=p["code_sha256_lf"])


def report(out,result):
    clean=pd.DataFrame([dict(method=m,**s) for m,s in result["metrics"].items()])
    cols=["method","precision","recall","f1","tn","fp","fn","tp","hard_fpr","no_evidence_n"]
    (out/"test_results.md").write_text("# Фінальне pairwise TEST-оцінювання\n\n"+
        table(clean,cols)+"\n\nConfusion matrix: [[TN,FP],[FN,TP]]. "
        "Пороги й моделі зафіксовані до TEST. Повні чисельності класів у JSON/CSV.\n",encoding="utf-8")
    masks=pd.read_csv(out/"missingness_results.csv")
    agg=masks.groupby(["scenario","missing_rate","method"])[["precision","recall","f1"]].agg(["mean","std"])
    agg.columns=["_".join(c) for c in agg.columns]
    agg=agg.reset_index()
    agg.to_csv(out/"missingness_summary.csv",index=False)
    (out/"missingness_analysis.md").write_text("# Стійкість до пропусків\n\n"+
        "Середнє та sample SD трьох offer-level масок. SD не є CI або оцінкою варіації незалежних датасетів. "
        "Пороги незмінні; title/identifier маскуються разом. Природні пропуски збережено.\n\n"+
        table(agg,list(agg.columns))+"\n",encoding="utf-8")
    pair=pd.DataFrame(result["paired_comparisons"])
    (out/"error_analysis.md").write_text("# Парні зміни помилок\n\n"+
        "Перехід from → to, на однакових повних TEST-парах. Мітки не змінювалися.\n\n"+
        table(pair,list(pair.columns))+"\n\nУсі scores, labels і decisions — test_predictions.csv; "
        "ідентифікатори, назви і чотири similarities збережені там само. "
        "Спільні лексичні помилки не виправляються після TEST.\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scenario in SCENARIOS:
        fig,ax=plt.subplots(figsize=(8,5))
        for m in METHODS:
            g=agg[(agg.scenario==scenario)&(agg.method==m)]
            ax.errorbar(g.missing_rate*100,g.f1_mean,yerr=g.f1_std,marker="o",label=m,capsize=3)
        ax.set(xlabel="Masked offers per field (%)",ylabel="Pairwise F1",ylim=(0,1),title=scenario)
        ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
        for ext in ("png","svg"):fig.savefig(out/f"f1_{scenario}.{ext}")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(clean.method,clean.f1);ax.set(ylabel="Pairwise F1",ylim=(0,1))
    ax.tick_params(axis="x",labelrotation=25);fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(out/f"clean_f1.{ext}")
    plt.close(fig)


def execute(c,bundle,train,pf):
    # No TEST bytes may be touched above this boundary.
    receipt=json.loads((OUT/"pretest_validation.json").read_text("utf-8"))
    if receipt["status"]!="PASSED" or receipt["code_sha256"]!=pf["code_sha256"] or receipt["config_sha256"]!=pf["config_sha256"]:
        raise ValueError("Missing or stale pre-TEST test receipt")
    with (OUT/"execution_started.json").open("x",encoding="utf-8") as f:
        json.dump(dict(status="STARTED",preflight=pf),f,indent=2)
    test=ROOT/"data/external"/TEST_NAME
    if digest(test)!=TEST_SHA:raise ValueError("TEST SHA mismatch; no parse performed")
    rows=read_pairs(test)
    audit,errors=validate_structure(rows,train)
    save(OUT/"test_data_audit.json",audit)
    if errors:raise ValueError("TEST integrity failure; no metrics: "+str(errors))
    y=np.array([int(p["label"]) for p in rows])
    hard=[p.get("is_hard_negative") for p in rows]
    with no_fit():
        features=bundle["extractor"].transform(rows);x=features.matrix
        values=score_all(x,bundle,c)
        preds={m:(values[m]>=c["final_models"][m]["threshold"]).astype(int) for m in METHODS}
        metrics={m:measures(y,preds[m],hard,np.isnan(x).all(axis=1)) for m in METHODS}
        records=[]
        for i,p in enumerate(rows):
            record={k:p.get(k) for k in ("pair_id","id_left","id_right","title_left","title_right","label","is_hard_negative")}
            record.update(coverage=float(evidence_coverage(x[i:i+1])[0]))
            for j,name in enumerate(("title","description","brand","identifier")):record["sim_"+name]=x[i,j]
            for m in METHODS:
                record[m+"_score"]=values[m][i];record[m+"_prediction"]=preds[m][i]
                record[m+"_threshold"]=c["final_models"][m]["threshold"]
            records.append(record)
        pd.DataFrame(records).to_csv(OUT/"test_predictions.csv",index=False,encoding="utf-8-sig")
        paired=[dict(from_method=a,to_method=b,**transitions(y,preds[a],preds[b]))
                for a,b in itertools.combinations(METHODS,2)]
        result=dict(status="CLEAN_SAVED",metrics=metrics,paired_comparisons=paired,
                    test_sha256=TEST_SHA,preflight=pf,configuration=c["final_models"])
        save(OUT/"test_results.json",result)
        pd.DataFrame([dict(method=m,**s) for m,s in metrics.items()]).to_csv(OUT/"test_metrics.csv",index=False)
        masked_rows=[]
        for scenario,rate,seed in itertools.product(SCENARIOS,RATES,SEEDS):
            xx=apply_missingness(features,rate,seed,scenario)
            vv=score_all(xx,bundle,c)
            removed=int((~np.isnan(x)&np.isnan(xx)).sum());original=int((~np.isnan(x)).sum())
            for m in METHODS:
                pred=(vv[m]>=c["final_models"][m]["threshold"]).astype(int)
                masked_rows.append(dict(scenario=scenario,missing_rate=rate,seed=seed,method=m,
                    removed_comparable_pair_fields=removed,original_comparable_pair_fields=original,
                    effective_removed_fraction=removed/original if original else 0.,
                    **measures(y,pred,hard,np.isnan(xx).all(axis=1))))
        pd.DataFrame(masked_rows).to_csv(OUT/"missingness_results.csv",index=False)
    result["status"]="COMPLETE"
    result["artifact_sha256"]={n:digest(OUT/n) for n in ("test_predictions.csv","test_metrics.csv","missingness_results.csv")}
    save(OUT/"test_results.json",result)
    report(OUT,result)
    save(OUT/"execution_completed.json",{"status":"COMPLETE","test_results_sha256":digest(OUT/"test_results.json")})
    print(json.dumps(metrics,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if (OUT/"execution_started.json").exists() or (OUT/"test_results.json").exists():
        raise SystemExit("Existing execution/results: verify saved artifacts; do not rerun TEST.")
    OUT.mkdir(parents=True,exist_ok=True)
    try:c,b,t,pf=preflight()
    except Exception as e:
        save(OUT/"preflight_failure.json",dict(status="FAILED",test_opened=False,error=str(e)))
        raise
    save(OUT/"preflight.json",pf)
    if args.execute:execute(c,b,t,pf)
    else:print("PREFLIGHT PASSED. TEST not opened.")


if __name__=="__main__":
    main()
