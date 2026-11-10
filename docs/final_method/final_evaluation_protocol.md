# Протокол майбутнього одноразового незалежного оцінювання

Зафіксовано **10 листопада 2026 року**, до нового corrected-fit і без відкриття зарезервованого TEST. Документ описує майбутній запуск, **не дозволяє його зараз**. Потрібне окреме явне завдання користувача після всіх перевірок frozen-конфігурації.

## Мета і дані

Основна задача — класифікація офіційно наданих WDC pairwise пар. Наперед визначено шість методів: TF-IDF title cosine, grouped Fixed Weighted, Renormalization, Adaptive v1, Logistic Regression, **Adaptive v1 corrected**. Головна метрика — binary F1; Precision, Recall, confusion matrix і hard-negative FPR подаються обов'язково. Corrected оцінюється незалежно від його development-місця.

Дозволене навчальне джерело:

- `data/external/wdcproducts80cc20rnd000un_train_small.json.gz`;
- SHA-256 `1915a92de76ddbf63a6c4d7ff3162df98f505033a8ad12ee7072b78b083d77c6`.

Майбутній TEST:

- `data/external/wdcproducts80cc20rnd100un_gs.json.gz`;
- очікуваний SHA-256 `258f9b408e715410f07480bf4ad39788a10d5cee1512e322091de43dafae1297`.

TEST-хеш узято **лише з уже наявного `docs/archive_manifest.json`**. Сам TEST зараз не читався, його новий хеш не обчислювався. Актуальний SHA перевіряється після окремого дозволу. Режим — 80cc/small TRAIN та 100un TEST: зміна сутностей не означає гарантованої якості перенесення.

Validation, TUNE, REPORT, FPV proposals і матеріали FPV-вебперевірок не додаються до фінального fit або tuning. Підозрілі official labels залишаються незмінними; після перегляду TEST заборонено вигідне одному методу relabel або видалення пар.

## Передумови готовності

1. Формула corrected остаточна: `S=C*M_q`, gamma `1`, prior `0`, identifier penalty `0`; групи, TRAIN-інформативність і ручна reliability успадковані.
2. Автотести, перевірка дозволених даних та збереження історичних файлів успішні.
3. `frozen_configuration.json` містить **одну** конфігурацію й один поріг на кожен із шести методів; outer-моделі не є набором для подальшого вибору.
4. Готовий `data/processed/final_method/frozen_models.joblib` містить fitted TRAIN extractor, інформативність і LR. Його SHA збігається з frozen JSON.
5. Збережені хеші TRAIN, коду, параметрів, специфікації, bundle; версії Python/пакетів, manifests і calibration provenance.
6. Final calibration B використовує лише inner-heldout scores; outer-рішення — лише development-опис; TEST не використовується.

CRLF/LF міграцію слід документувати окремо: для Python-коду допускається чітко визначений LF-нормалізований SHA разом із фактичним raw SHA. Інші зміни не прирівнювати до перенесення без перевірки. Для бінарних даних і bundle SHA побайтовий. Зміну бібліотек чи коду перевіряти **без TEST**, а не після фінальних метрик.

## Порядок навчання до TEST

1. TRAIN ділиться за незмінним 3×3 entity-group manifest, seed `20261101`. Пара належить частині тільки за обома endpoints; crossing negatives виключені й обліковані.
2. У development A corrected навчає TF-IDF та інформативність заново в кожному inner fit; поріг вибирається за відповідними pooled inner OOF. Outer development не визначає нових параметрів.
3. У final calibration B для кожної candidate-конфігурації збираються тільки дев'ять INNER heldout score-наборів. Повторні оригінальні pair ID отримують середній score та одну рівновагову мітку; multiplicity збережено. Порядок вибору: F1, Precision, вищий поріг, початковий grid order.
4. Corrected має лише поріг `.01..99`. Для baseline сітки незмінні: v1 gamma `.5/1/1.5`, prior `.20`, penalty `0`; LR C `.1/1/10`, L2, lbfgs, class_weight=None, max_iter=2000; інші методи мають лише поріг. Розширювати сітки після перегляду результатів заборонено.
5. Один extractor навчається на всіх TRAIN-пропозиціях із дедуплікацією offer ID; інформативність — на TRAIN similarities/labels; одна LR — на тих самих TRAIN similarity/availability features з вибраним C. Артефакт і всі шість порогів заморожено до TEST.

Середній inner-calibration score не є ансамблем у bundle. Full-TRAIN refit змінює class prior, IDF і шкалу score; переносимість порога є прийнятим обмеженням. Виправляти її за TEST або обирати іншу модель після фінальних метрик не можна.

## Одноразовий запуск після окремого дозволу

1. До парсингу TEST завантажити frozen JSON, перевірити bundle/specification/code/package hashes і статус готовності. Невідповідність або відсутність артефакта — причина зупинитись, а не fit на TEST.
2. Звірити TEST filename і побайтовий SHA. Лише після збігу прочитати official файл у межах одного дозволеного фінального процесу.
3. Перевірити схему, унікальність offer/pair IDs, узгодженість повторів, labels і cluster metadata. Перетини TRAIN/TEST offer IDs, сутностей і пар перевірити за метаданими. Однакові нормалізовані тексти облікувати; вони самі по собі не доводять витоку. Несподівана структурна проблема зупиняє запуск до метрик; пари не фільтруються заднім числом.
4. Виконувати **тільки transform** готового TRAIN extractor. Не доповнювати vocabulary/IDF та не перераховувати інформативність, частоти, scaler, imputer, class weights або LR на TEST.
5. Усі шість методів оцінюють ті самі повні official TEST-пари з frozen-параметрами. Позитивне рішення: `score >= threshold`. Показати всі шість результатів. Candidate generation не застосовується: воно змінило б denominator pairwise benchmark.
6. Зберегти один основний JSON, таблицю метрик, конфігурацію, per-pair scores/рішення й хеші до змістового аналізу помилок. Не перезаписувати невдалий результат іншим порогом і не приховувати погіршення corrected.

Технічне повторення після збою можливе лише з тим самим кодом/даними/артефактом/конфігурацією та документованою причиною; це не нова спроба вибрати кращий F1. Після перегляду TEST він стає переглянутим набором. Подальші зміни потребують нового незалежного підтвердження.

## Невідомі сутності й пропуски

Unseen сутність не відкидається через відсутність cluster ID у TRAIN. IDs/cluster IDs використовуються для маршрутизації та аудиту, не як model features. Новий текст нормалізується й трансформується замороженим словником. Порожній/OOV вектор означає недоступну cosine (`NaN`); спостережуваний нуль залишається нулем.

Відсутні характеристики не доповнюються з іншої пропозиції або TEST-корпусу. Identifier залишається тією самою евристикою; нерівність brand чи нульовий Jaccard не є автоматично доведеною суперечністю manufacturer MPN. Title/identifier мають один бюджет. Нові технічні FPV-ознаки в цьому WDC-протоколі не додаються.

За всіх відсутніх каналів TF-IDF/Fixed/Renormalization/v1/corrected мають score `0`; позитивний поріг дає nonmatch. LR використовує similarity `0`, availability `0` і **навчений intercept**. Post-hoc all-missing rejection для LR не додається. Окремо записати кількість all-missing і позитивних рішень серед них.

## Основні й додаткові метрики

Для кожного методу:

- Precision `TP/(TP+FP)`, Recall `TP/(TP+FN)`, F1 `2TP/(2TP+FP+FN)`; нульовий знаменник дає `0`;
- `N`, positives/negatives і `TN/FP/FN/TP`;
- confusion matrix саме `[[TN,FP],[FN,TP]]`.

На official hard negatives (`label=0`, `is_hard_negative=True`) та решті негативів окремо: N, FP, TN, FPR `FP/(FP+TN)`, specificity. F1 negative-only підмножини не інтерпретується як якість hard negatives. Hard-прапорець — reporting metadata, не feature. Якщо прапорець відсутній, записати відсутність, не вигадувати lexical hard labels.

Показати парні відмінності corrected від Fixed/Renormalization/v1: кількість різних рішень, виправлених і доданих FP/FN. Приріст F1 сам по собі не доводить фундаментальної новизни, калібрації чи статистично доведеної переваги. Scores — виходи алгоритмів.

## Наперед задана secondary-перевірка пропусків

Після незмінної clean-таблиці у тому самому дозволеному процесі, без нового fit:

- `auxiliary`: description/brand;
- `all_groups`: title разом з identifier, description, brand.

Рівні `0%,10%,20%,30%,50%`, seeds `11,22,33`. Один deterministic SHA256 mask на offer ID і сире поле, спільний для всіх пар і методів; title/identifier приховуються разом. Природні пропуски зберігаються. Всі моделі й пороги frozen на clean TRAIN.

На scenario/rate/seed записати P/R/F1, confusion matrix, фактичну втрату доступних similarity-каналів і no-evidence counts/positives. Середнє та вибіркове SD трьох масок описують маскування, не незалежні датасети або довірчий інтервал. Clean-метрики основні; за масками не добирати іншу формулу чи поріг.

## Pairwise, candidates і end-to-end пошук

WDC TEST дає **pairwise** Precision/Recall/F1 на наданих парах. Він не вимірює повноту TF-IDF candidate generation і не є end-to-end search benchmark.

Candidate recall потребує вичерпного переліку релевантних пар у цільовому каталозі та зафіксованої генерації кандидатів. Recall@K/Precision@K потребують query-wise relevance gold для всього відповідного каталогу, правил інших магазинів і визначених запитів. Неповні WDC pairs та непідтверджені FPV proposals цього не дають; такі метрики за цим протоколом не обчислюються.

Демонстрація `--method adaptive-corrected` на WDC TRAIN показує лише інтеграцію готового scorer. CLI default залишається Renormalization. Подібність результату не підтверджує тотожність товару або якість реального вебпошуку.
