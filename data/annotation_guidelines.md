# Optional supplemental dataset: electronic motors — annotation protocol

**Pilot available; no gold labels yet.** As of 3 November 2026 the local FPV pilot contains 32 offers and 7 unconfirmed candidate pairs. See the [dataset audit](../docs/fpv_dataset_audit.md). The initial `motor_offers_template.csv` below is a separate collection template, not the current pilot schema.

1. Unit of observation: one real seller offer URL captured on a known date. Keep the original product name and raw attributes; avoid paraphrasing them to match other sellers.
2. Unit of identity: one manufacturer model **including material variants** (such as winding/KV, dimension, and revision). Functional compatibility is not a positive match.
3. Ground truth: assign the same `product_entity_id` only after checking an independent manufacturer model code/specification or another verifiable identity source. Store the evidence URL. Where identity remains ambiguous, use `review_status=uncertain` and exclude from formal test.
4. Keep near-identical but different revisions and KV variants as **hard negative** candidates. Report the selection strategy; do not make the evaluation consist solely of easy negatives.
5. Do not fill missing seller fields from another offer; that would artificially remove the natural missingness being studied. A verified reference specification may support an identity decision but must be stored separately from each seller's observed attributes.
6. Record manufacturer, exact model, KV, stator dimensions, voltage and weight where present, preserving raw units. Use blank for absent values. Distinguish `missing` from extraction error; verify extracted numbers against the source.
7. Prioritize repeated exact variants across multiple retailers and more independent model families. The initial ~80–120 offers / 25–40 models are planning targets, not a statistically justified sample size. Include independent second review of annotations; the present three potential cross-store positive groups are insufficient for a reliable train/validation/test evaluation.
8. Prevent leakage: for a dedicated hold-out, keep product entities used in test out of any training/validation split. Use WDC for primary independent benchmark evaluation if this domain sample is too small for reliable parameter tuning.
9. Generate candidate pairs and *evaluate candidate recall separately* if claiming end-to-end search recall. For pairwise classification, positives and negatives must come from the same documented candidate procedure.
10. Record reasons for exclusion, unknown model identifiers and conflicts. Document scraping/robots/terms restrictions and avoid disallowed collection methods.

Template file: `motor_offers_template.csv` (header only; no products or fictional data).

## Ручна перевірка поточного пілоту

`data/external/fpv_motors/review_queue.csv` містить усі 7 пар. `proposed_label_1_match_0_nonmatch`, `verification_status`, `rationale` і поля пропозицій `_a`/`_b` є вихідними даними; їх не редагувати. Початковий `review_status=pending`, підтверджена мітка порожня.

- `confirmed`: перевірку завершено; заповнити `confirmed_label_1_match_0_nonmatch` точним `0`/`1`, `reviewer`, `reviewed_on=YYYY-MM-DD`, `evidence_url` та пояснення в `review_notes`. Запропонована мітка може виявитися неправильною.
- `uncertain` або `excluded`: мітка залишається порожньою; записати причину в `review_notes`/`exclusion_reason`. Самих однакових KV і статора недостатньо для підтвердження.
- `variant_resolution` описує перевірку модифікації, але не замінює виправлення неоднозначного вихідного запису. DR05/DR06 мають бути розв'язані за власним джерелом до оцінювання. Не створювати штучні offers розділенням неперевіреної картки.

До анотування погодити одиницю тотожності: рекомендовано модифікацію двигуна з істотними KV/розміром/ревізією; кількість у наборі та додаткові компоненти фіксувати окремо. Exact sale SKU є іншою ціллю. Поточні примітки про порівняння 4-pack тільки з 4-pack потребують такого узгодження; single/4-pack не є автоматичною міткою незбігу моделі.

`seller_sku` локальний для магазину. FT/BrotherHobby та AOS RC/RCINPOWER не об'єднувати як aliases без джерельних свідчень. Зберігати початкові коди й одиниці, зокрема `0802`, `2207.5`, `22.6-6.6`; пропуски не доповнювати за іншою пропозицією. CW-пропелер у PH04 не встановлює напрям обертання двигуна.

Повторний аудит зберігає ручні поля й блокує перезапис за зміни джерел/складу пар. Для документованого виправлення даних зберегти попередню версію та створити нову чергу в іншому файлі. В електронній таблиці імпортувати коди й ID як текст. Формат URL свідчення перевіряється програмно; достовірність свідчення перевіряє рецензент.

Для майбутнього імпорту підтверджених міток потрібен окремий експорт тільки рядків `confirmed` із повною схемою. Змішана черга відхиляється. Якщо FPV-дані використано для зміни методу, вони стають development-даними; для підтвердження потрібен новий незалежний hold-out. Поточний етап не передбачає метрик чи фінальних експериментів.
