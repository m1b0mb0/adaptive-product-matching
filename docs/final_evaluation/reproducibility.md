# Відтворюваність фінального оцінювання

Авторизовано користувачем 16 листопада 2026 року. Актуальний workspace:
D:/Development/masters/adaptive-product-matching. Попередніх фінальних
результатів у локальних docs/results/artifacts/data/processed не знайдено.

Джерело істини: незмінні docs/final_method/frozen_configuration.json,
final_evaluation_protocol.md, method_specification.md, implementation_status.md.
Python і пакети, TRAIN/bundle/config/code hashes — preflight.json.
Перед TEST збережено pretest_validation.json, включно з SHA runner і тестів.
Історичні code hashes нормалізовані за LF; нові runner hashes — побайтові.

Процедура до TEST:
1. Метрики перевірено проти scikit-learn на контрольних двокласових,
   однокласових і порожніх фікстурах. Для порожнього випадку задана
   нульова матриця вручну: поточний sklearn відхиляє порожній вхід.
2. Frozen bundle перевірено за SHA, конфігурацією, LR metadata,
   representation metadata, інформативністю та TRAIN SHA.
3. На TRAIN smoke та фікстурах перевірено transform, OOV, missing,
   observed zero, шість методів і заборону fit.
4. Генерація MD/CSV і графіків перевірена у тимчасовому каталозі.
5. Успішні 13 тестів, потім preflight і запис pretest receipt.

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_final_evaluation.py --junitxml=tmp/final_evaluation_pretest.xml
.\.venv\Scripts\python.exe -X utf8 final_evaluation.py
# Після збереження успішного pretest receipt, рівно один дозволений запуск:
.\.venv\Scripts\python.exe -X utf8 final_evaluation.py --execute
~~~

Runner створює exclusive execution_started.json до читання TEST.
Наявність цього маркера або test_results.json блокує повторний запуск.
Не видаляти маркери для повторної спроби; технічний збій потребує
документованого розбору з тими самими моделями/кодом/даними.

Після preflight перевіряється SHA офіційного 100un TEST. Всі official
пари проходять схему, дублікати, label/cluster consistency та overlap
із TRAIN. Несподівана цілісність зупиняє процес до метрик; фільтрації
пар за помилками немає. Однакові тексти обліковуються, але не є
автоматичним доказом витоку. Нові сутності не відкидаються.

Scoring: лише frozen transform; score >= точний поріг із JSON.
Відсутність свідчень: нуль для евристик, fitted intercept для LR.
Відомий нуль подібності відрізняється від NaN. Candidate retrieval
не застосовується. Runtime guard блокує fit основних компонентів.

Clean per-pair scores, predictions, labels, similarities, thresholds
записуються у test_predictions.csv; метрики — test_metrics.csv і JSON.
Ці результати зберігаються до аналізу. Потім лише predeclared masks:
auxiliary/all_groups, 0/.1/.2/.3/.5, seeds 11/22/33.
SHA256(seed:field:offer_id) задає однакову маску у всіх парах і методах.
Маски можна незалежно перевірити зі збереженої clean feature matrix
та offer IDs; прихованих перетворень, refit або нових thresholds немає.

JSON містить SHA числових артефактів. Незалежна перевірка має читати
збережені CSV й відновлювати confusion counts, а не запускати TEST знову.
Змістову інтерпретацію доповнюють після збереження результатів;
код оцінювання після TEST не змінюється.


## Завершення

Один фінальний процес завершено успішно. Кількість пар — 4500.
Збережено execution_started.json і execution_completed.json;
технічного повторення TEST не було.

Незалежна перевірка за збереженим CSV звірила всі шість confusion
matrices та P/R/F1 зі sklearn, hard-negative counts, усі парні
переходи, score>=threshold, 180 missingness-рядків, rate-zero parity,
спільні mask counts і sample SD. Результат:
independent_verification.json.

Перевірено незмінність 149 історичних файлів, включно з frozen bundle,
та незмінність SHA runner і тестів відносно pretest receipt.
Первинні TEST JSON/CSV не редагувалися після збереження; до MD
додано лише інтерпретацію. Графіки побудовано з цих самих таблиць.
