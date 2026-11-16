# Парні зміни помилок

Перехід from → to, на однакових повних TEST-парах. Мітки не змінювалися.

| from_method | to_method | disagreements | corrected_fp | added_fp | corrected_fn | added_fn |
| --- | --- | --- | --- | --- | --- | --- |
| text | fixed | 177 | 123 | 45 | 4 | 5 |
| text | renorm | 239 | 212 | 13 | 3 | 11 |
| text | adaptive | 173 | 87 | 80 | 4 | 2 |
| text | logistic | 1596 | 1381 | 0 | 2 | 213 |
| text | adaptive_corrected | 166 | 118 | 40 | 3 | 5 |
| fixed | renorm | 128 | 121 | 0 | 0 | 7 |
| fixed | adaptive | 112 | 18 | 89 | 4 | 1 |
| fixed | logistic | 1513 | 1303 | 0 | 0 | 210 |
| fixed | adaptive_corrected | 21 | 10 | 10 | 0 | 1 |
| renorm | adaptive | 202 | 0 | 192 | 10 | 0 |
| renorm | logistic | 1385 | 1182 | 0 | 0 | 203 |
| renorm | adaptive_corrected | 127 | 0 | 121 | 6 | 0 |
| adaptive | logistic | 1587 | 1374 | 0 | 0 | 213 |
| adaptive | adaptive_corrected | 91 | 79 | 8 | 0 | 4 |
| logistic | adaptive_corrected | 1512 | 0 | 1303 | 209 | 0 |

Усі scores, labels і decisions — test_predictions.csv; ідентифікатори, назви і чотири similarities збережені там само. Спільні лексичні помилки не виправляються після TEST.


## Fixed, Renorm і corrected

- Fixed → Renorm: 128 різних рішень; 121 FP усунуто, 7 FN додано.
- Fixed → corrected: 21 різне рішення; 10 FP усунуто, 10 FP додано,
  1 FN додано.
- Renorm → corrected: 127 різних рішень; 6 FN усунуто ціною 121
  додаткового FP; виправлених FP немає.
- v1 → corrected: 91 різне рішення; 79 FP усунуто, 8 FP додано,
  4 FN додано.

Повний перелік змінених пар цих порівнянь — decision_changes.csv.
Ці переходи описують зафіксовані operating points, не тільки
властивості формул окремо від порогів.

## Приклади зі збережених результатів

Детерміновано перші три LR FP за pair_id:
1000748#21940746 — Skout's Honor Detangling Spray / Shampoo;
10132766#3713378 і 10171935#10132766 — Omega Diver 300M /
Planet Ocean 600M. Спільні бренд/сімейство не відокремлюють
тип товару або модифікацію. Це приклади, не повна анотація причин.

Перші три LR FN:
10132766#65287375 — коротка Omega Diver 300M назва /
довга назва з GMT Chronograph і кодом;
10330385#44917290 — Railmaster / Replica Seamaster;
10816#20448379 — BlueSolar Charge Controller / Remote Panel.

Останні два приклади також підкреслюють потребу обережного
тлумачення official gold: назви виглядають неоднозначно або описують
різні товари. Тут немає джерельної перевірки; labels залишено,
пари не вилучено, алгоритми не адаптовано до цих прикладів.
