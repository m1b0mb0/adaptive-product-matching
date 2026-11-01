# Optional supplemental dataset: electronic motors — annotation protocol

**Not available yet.** This document defines how to collect evidence; it is not a claim that data have been collected.

1. Unit of observation: one real seller offer URL captured on a known date. Keep the original product name and raw attributes; avoid paraphrasing them to match other sellers.
2. Unit of identity: one manufacturer model **including material variants** (such as winding/KV, dimension, and revision). Functional compatibility is not a positive match.
3. Ground truth: assign the same `product_entity_id` only after checking an independent manufacturer model code/specification or another verifiable identity source. Store the evidence URL. Where identity remains ambiguous, use `review_status=uncertain` and exclude from formal test.
4. Keep near-identical but different revisions and KV variants as **hard negative** candidates. Report the selection strategy; do not make the evaluation consist solely of easy negatives.
5. Do not fill missing seller fields from another offer; that would artificially remove the natural missingness being studied. A verified reference specification may support an identity decision but must be stored separately from each seller's observed attributes.
6. Record manufacturer, exact model, KV, stator dimensions, voltage and weight where present, preserving raw units. Use blank for absent values. Distinguish `missing` from extraction error; verify extracted numbers against the source.
7. Aim for ~80–120 offers spanning 25–40 distinct models and multiple retailers **only if collecting them is feasible by 5 November**; these are planning targets, not claims about sampling sufficiency. Include a review sample to estimate label error.
8. Prevent leakage: for a dedicated hold-out, keep product entities used in test out of any training/validation split. Use WDC for primary independent benchmark evaluation if this domain sample is too small for reliable parameter tuning.
9. Generate candidate pairs and *evaluate candidate recall separately* if claiming end-to-end search recall. For pairwise classification, positives and negatives must come from the same documented candidate procedure.
10. Record reasons for exclusion, unknown model identifiers and conflicts. Document scraping/robots/terms restrictions and avoid disallowed collection methods.

Template file: `motor_offers_template.csv` (header only; no products or fictional data).
