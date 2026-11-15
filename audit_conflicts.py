"""Audit textual conflict candidates on cached TRAIN predictions; no penalties."""
import json
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd
from matching import identifiers, norm
from study_method_selection import checked_cache, align_predictions, sha, table

OUT = Path("docs/conflict_aware_study")
SIZE = re.compile(r'(?<![\w.])(\d+(?:[.,]\d+)?)\s*(mm|cm|inches|inch|in\.?)(?!\w)')
LABELLED = re.compile(r'\b(?:mpn|model(?:\s+(?:number|no\.?))?)\s*[:=]\s*([a-z0-9][a-z0-9._-]*)')
UNITS = {"mm":"mm","cm":"cm","inch":"inch","inches":"inch","in":"inch","in.":"inch"}


def sizes(text):
    """Lexical values only: dimensions/roles are NOT inferred or certified."""
    text = norm(text)
    found, ambiguous = [], False
    for m in SIZE.finditer(text):
        prefix = text[:m.start()].rstrip()
        # Reject range/ratio/dimension-chain suffix masquerading as one value.
        if re.search(r"(?:\d|mm|cm|inch)\s*[-–—/:x×]$", prefix):
            ambiguous = True
        found.append((UNITS[m[2]], str(Decimal(m[1].replace(",", ".")).normalize())))
    values = sorted(set(found))
    if len(values) > 1:
        ambiguous = True
    return dict(values=values, ambiguous=ambiguous)


def dimension_status(left, right):
    a,b=sizes(left),sizes(right)
    if a["ambiguous"] or b["ambiguous"]:
        return "ambiguous"
    if not a["values"] or not b["values"]:
        return "missing_both" if not a["values"] and not b["values"] else "missing_one"
    (u,x),(v,y)=a["values"][0],b["values"][0]
    if u!=v:
        return "different_units"
    return "same_value_unverified_role" if x==y else "different_value_unverified_role"


def code_status(left,right):
    a,b=identifiers(left),identifiers(right)
    if not a or not b:
        return "missing_both" if not a and not b else "missing_one"
    if not a & b:
        return "disjoint_singletons" if len(a)==len(b)==1 else "disjoint_multiple"
    return "equal_sets" if a==b else "partial_overlap"


def labelled_codes(text):
    return sorted({m[1] for m in LABELLED.finditer(norm(text))
                   if re.search("[a-z]",m[1]) and re.search("[0-9]",m[1])})


def audit(data):
    rows=[]
    for p in data.to_dict("records"):
        a,b=p["title_left"],p["title_right"]
        ca,cb=labelled_codes(a),labelled_codes(b)
        rows.append({
            **{k:p[k] for k in ["pair_id","outer_fold","label","is_hard_negative",
                "title_left","title_right","sim_identifier","sim_title",
                "fixed_prediction","renorm_prediction","renorm_score","renorm_threshold"]},
            "identifier_left":json.dumps(sorted(identifiers(a))),
            "identifier_right":json.dumps(sorted(identifiers(b))),
            "identifier_status":code_status(a,b),
            "dimension_left":json.dumps(sizes(a)),
            "dimension_right":json.dumps(sizes(b)),
            "dimension_status":dimension_status(a,b),
            "labelled_codes_left":json.dumps(ca),
            "labelled_codes_right":json.dumps(cb),
            "both_explicit_code_markers":bool(ca and cb),
            # No manufacturer authority or attribute-role gold exists in cache.
            "verified_conflict":False,
            "common_fp":p["label"]==0 and p["fixed_prediction"]==p["renorm_prediction"]==1,
            "renorm_tp":p["label"]==1 and p["renorm_prediction"]==1,
        })
    return pd.DataFrame(rows)


def summarize(rows):
    result=[]
    for field in ["identifier_status","dimension_status","both_explicit_code_markers"]:
        for value,g in rows.groupby(field,sort=True):
            negatives=g.label.eq(0)
            result.append(dict(signal=field,status=value,n=len(g),
                positives=int(g.label.eq(1).sum()),negatives=int(negatives.sum()),
                hard_negatives=int((negatives & g.is_hard_negative.astype(str).str.lower().eq("true")).sum()),
                common_fp=int(g.common_fp.sum()),renorm_fp=int((negatives & g.renorm_prediction.eq(1)).sum()),
                renorm_tp_at_risk=int(g.renorm_tp.sum()),
                renorm_existing_fn=int((g.label.eq(1)&g.renorm_prediction.eq(0)).sum())))
    return pd.DataFrame(result)


def main():
    root=Path(__file__).resolve().parent
    frame,result,provenance=checked_cache(root)
    data=align_predictions(frame,result)
    rows=audit(data)
    summary=summarize(rows)
    out=root/OUT
    out.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out/"pair_audit.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(out/"signal_summary.csv",index=False,encoding="utf-8-sig")
    # Review EVERY candidate, not just baseline errors or gold negatives.
    candidates=rows[rows.identifier_status.str.startswith("disjoint")|
                    rows.dimension_status.eq("different_value_unverified_role")|
                    rows.both_explicit_code_markers]
    candidates.to_csv(out/"review_candidates.csv",index=False,encoding="utf-8-sig")
    (out/"audit_tables.md").write_text(
        "# Описові зрізи TRAIN development\n\n"
        "Це кандидати для аудиту, не підтверджені суперечності. "
        "TP at risk означає кількість поточних TP серед спрацьовувань, "
        "а не виміряну втрату Recall від штрафу. Групи різних сигналів перетинаються.\n\n"+
        table(summary,list(summary.columns))+"\n",encoding="utf-8")
    report=dict(scope="cached TRAIN development titles only",
        pairs=len(rows), positives=int(rows.label.sum()),
        common_fp=int(rows.common_fp.sum()), model_fits=0, penalties_applied=0,
        thresholds_selected=0,test_opened=False,fpv_labels_used=False,
        source_sha256=provenance,diagnostic_script_sha256=sha(Path(__file__)),
        verified_conflicts="not established: no manufacturer provenance or attribute-role gold")
    (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(summary.to_string(index=False))


if __name__=="__main__":
    main()
