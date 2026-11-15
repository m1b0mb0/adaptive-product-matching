# Описові зрізи TRAIN development

Це кандидати для аудиту, не підтверджені суперечності. TP at risk означає кількість поточних TP серед спрацьовувань, а не виміряну втрату Recall від штрафу. Групи різних сигналів перетинаються.

| signal | status | n | positives | negatives | hard_negatives | common_fp | renorm_fp | renorm_tp_at_risk | renorm_existing_fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identifier_status | disjoint_multiple | 114 | 35 | 79 | 31 | 25 | 26 | 34 | 1 |
| identifier_status | disjoint_singletons | 144 | 28 | 116 | 68 | 53 | 57 | 24 | 4 |
| identifier_status | equal_sets | 87 | 79 | 8 | 8 | 8 | 8 | 79 | 0 |
| identifier_status | missing_both | 285 | 139 | 146 | 82 | 71 | 72 | 137 | 2 |
| identifier_status | missing_one | 428 | 157 | 271 | 92 | 71 | 71 | 154 | 3 |
| identifier_status | partial_overlap | 77 | 62 | 15 | 15 | 15 | 15 | 62 | 0 |
| dimension_status | ambiguous | 45 | 14 | 31 | 16 | 14 | 15 | 13 | 1 |
| dimension_status | different_units | 4 | 2 | 2 | 0 | 0 | 0 | 2 | 0 |
| dimension_status | different_value_unverified_role | 16 | 1 | 15 | 12 | 12 | 12 | 1 | 0 |
| dimension_status | missing_both | 942 | 423 | 519 | 251 | 204 | 208 | 414 | 9 |
| dimension_status | missing_one | 98 | 33 | 65 | 14 | 10 | 11 | 33 | 0 |
| dimension_status | same_value_unverified_role | 30 | 27 | 3 | 3 | 3 | 3 | 27 | 0 |
| both_explicit_code_markers | False | 1135 | 500 | 635 | 296 | 243 | 249 | 490 | 10 |
