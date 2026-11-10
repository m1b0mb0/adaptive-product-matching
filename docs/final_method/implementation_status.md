# Стан реалізації та готовність до окремого фінального оцінювання

Стан на **10 листопада 2026 року**. Поточний підтверджений workspace: `D:\Development\masters\adaptive-product-matching`. Команди й локальні посилання в цьому документі задаються відносно цього кореня. Історичні абсолютні шляхи не визначають актуальний проєкт.

## Відновлене середовище та дані

Python **3.12.10**, ізольоване середовище `.venv`. Після перенесення Python був доступний, але наукові пакети в `.venv` були відсутні; їх відновлено за незміненим `requirements-lock.txt`. Основні версії збігаються з історичним development provenance:

| Пакет | Версія |
| --- | --- |
| NumPy | 2.5.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| matplotlib | 3.11.2 |
| pytest | 9.2.0 |

Усі обов'язкові файли проєкту й `docs/known_methods/` знайдено. Наявність зарезервованого TEST встановлено лише за переліком файлів; його вміст і новий SHA у цьому етапі **не читалися**.

Перевірені дозволені локальні джерела:

| Файл | SHA-256 |
| --- | --- |
| `data/external/wdcproducts80cc20rnd000un_train_small.json.gz` | `1915a92de76ddbf63a6c4d7ff3162df98f505033a8ad12ee7072b78b083d77c6` |
| `data/external/wdcproducts80cc20rnd000un_valid_small.json.gz` | `892e8d39cc8230dce5039a4bded16be1237d5c5c37b94a37404226d349ef3df8` |
| `data/external/fpv_motors/fpv_motors_offers_pilot.csv` | `9a96914af39ed19514b1f547501a37866cf4352dc2f5425517acead0a290b85f` |
| `data/external/fpv_motors/fpv_motors_pair_candidates.csv` | `361299546b911b99a8ca39ad639fb37ffdf7473d9e7263acef85fbce247b4e8a` |

TRAIN, validation та FPV SHA узгоджені з попередніми артефактами. Validation перевірено як відновлений файл, але не використано для нового fit/tuning. FPV-файли й черга збережені; датасет не розширювався, непідтверджені мітки не використовувалися.

Після міграції root Python-файли мають CRLF, тоді як історичні SHA були обчислені для LF. Для п'яти baseline-модулів LF-нормалізовані SHA точно збігаються з попереднім provenance; raw SHA записано окремо. Це перевірена різниця закінчень рядків, а не зміна алгоритмів. `matching.py` залишено без функціональних або текстових змін цього етапу. П'ять файлів immutable snapshot, для яких CRLF порушив старі побайтові перевірки, відновлено до точних історичних LF-байтів після перевірки відмінності лише в закінченнях рядків. Історичні результати не перераховувалися й не переписувалися.

## Реалізовані компоненти

| Компонент | Стан |
| --- | --- |
| `adaptive_corrected.py` | Один фіксований `S=C*M_q`, gamma=1, prior=0, identifier penalty=0; успадковані групи й ручна reliability |
| `FrozenCorrectedScorer` | Готові extractor/інформативність, лише transform/score після створення |
| `prepare_final_method.py` | Corrected nested development A, inner-only final calibration B, один full-TRAIN bundle |
| `frozen_method_artifact.py` | Перевірка погодженого локального bundle й конфігурації, без query-time fit |
| `search_system.py`, `search_cli.py` | Окремий `adaptive-corrected`; наявний default Renormalization збережено |
| Історичні методи та FPV-адаптер | Збережені; v2/v3, нові ML-моделі й додаткова інфраструктура не реалізовані |

Ручні `r=[1,1,1,0.70]` не проголошуються емпірично встановленою надійністю джерел. Інформативність оцінено лише на дозволених TRAIN fit-частинах, а фінальні фактори на всьому TRAIN: `[1.0567295966684407, 0.8108932609368653, 1.5, 1.4083394342585518]` у порядку title/description/brand/identifier.

## Development та одна фінальна конфігурація

П'ять попередніх baseline-результатів повторно використано після перевірки даних, core-code LF SHA, точних версій, split manifest, артефактів і метрик CSV. Старий nested baseline-експеримент не запускався знову. Нові fit необхідні для corrected і для ще не існуючої єдиної фінальної calibration-конфігурації.

Corrected оцінено на тих самих **1135 outer development-парах**: 500 позитивних і 635 негативних; 1365 crossing negatives виключено. Corrected отримав Precision **0.659517**, Recall **0.984000**, F1 **0.789727**, confusion matrix `[[381,254],[8,492]]`. Це розвідувальні результати, не незалежне підтвердження. Вони не перевершують найкраще історичне clean development F1; формулу після їх перегляду не змінено. Повна таблиця — у [development_comparison.md](development_comparison.md).

Final calibration B використовує лише дев'ять **inner-heldout** наборів. Повторні scores усереднено за оригінальною парою; fold-пороги не усереднювалися, outer-рішення для нового добору не застосовувалися. Отримано **1203 унікальні пари: 500 позитивних і 703 негативних**. Multiplicity: 729 пар мали два inner-heldout scores, 474 — один; після згортання кожна пара має рівну вагу. Selection-метрики не є незалежними показниками якості.

Після calibration виконано один full-TRAIN fit. Зафіксовані параметри:

| Метод | Поріг | Інші параметри |
| --- | ---: | --- |
| TF-IDF | 0.16 | title-only |
| Fixed Weighted | 0.11 | наявні групові бюджети |
| Renormalization | 0.16 | наявні групові бюджети |
| Adaptive v1 | 0.14 | gamma=1, prior=0.20, penalty=0 |
| Logistic Regression | 0.29 | C=0.1, L2/lbfgs |
| Adaptive v1 corrected | 0.11 | gamma=1, prior=0, penalty=0 |

Таблиця округлена для читання. Виконувані точні floating-point значення, специфікація, сітки, версії та provenance містяться у [frozen_configuration.json](frozen_configuration.json).

- SHA-256 параметрів: `65fb3492f67c27735f7280047992563d1efebf806b10aa356feb7b53326a9556`.
- Bundle: `data/processed/final_method/frozen_models.joblib`.
- SHA-256 bundle: `329fa0919f570172956b3fe60aa97dce68357f8189370ea317f3221487ff5caa`.

## Пошукова інтеграція та перевірки

Реальний corrected-приклад на WDC TRAIN виконано успішно: каталог 1000 пропозицій, 20 TF-IDF-кандидатів, 5 результатів; першим повернуто `84668470`, score `0.37695818679718324`. Це не перевірена відповідність і не незалежна оцінка пошуку. Виводи — `search_demo.json` та `search_demo.txt`.

```powershell
.\.venv\Scripts\python.exe -X utf8 search_cli.py --catalog-format wdc --catalog data/external/wdcproducts80cc20rnd000un_train_small.json.gz --query-id 1017836 --candidate-k 20 --top 5 --method adaptive-corrected --method-config docs/final_method/frozen_configuration.json
```

Corrected-параметри не навчаються на запиті. Його vocabulary/IDF та інформативність беруться з готового TRAIN bundle. Окремий catalog TF-IDF для retrieval будується як індекс, без міток; він не замінює заморожений extractor парного scorer. Поріг тотожності не використовується для ранжування. Наявний in-memory LR hook і стандартні режими пошуку збережено.

**483 тести пройшли.** Перевірено all-missing, NaN проти observed zero, один бюджет title/identifier, bounds, відсутність prior, незмінність v1, відтворюваність, fold-local fit, inner-only calibration, frozen artifact і search-інтеграцію. Тестові fixtures є перевірками програмного забезпечення, не штучними науковими спостереженнями.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Збереження та відтворення після перенесення

`data/external/` і `data/processed/` виключені `.gitignore`. Потрібно зберегти сирі джерела й **оригінальний model bundle** окремо та після перенесення звірити їхні SHA із frozen JSON. Наявність лише коду й JSON без bundle не означає, що corrected-пошук готовий. За відсутності або невідповідності артефакта пошук завершується помилкою; прихованого fit на запиті немає.

У разі документованої необхідності відтворення, без TEST та без перезапису історичних результатів:

```powershell
.\.venv\Scripts\python.exe -X utf8 prepare_final_method.py --out-dir docs/final_method/reproduction
```

Для reproduction використовується окремий namespace артефакта. Порівнювати потрібно параметри, метод, маніфести й контрольні суми з оригінальним frozen-конфігом; не вибирати «кращий запуск» за F1. Оригінальний frozen JSON і bundle не підміняються мовчки. Якщо відтворення не дає тих самих параметрів, причину досліджують без TEST. Для точного перенесення збереженого артефакта його побайтовий SHA має збігатися; повторна серіалізація може створити інші байти та потребує окремого provenance.

## Залишкові обмеження та наступний дозвіл

- TRAIN/TUNE/REPORT уже переглядалися, тому новий development не є незалежним підтвердженням.
- Видалення crossing negatives змінює баланс; після full-TRAIN refit змінюються class prior, IDF і score scale. Переносимість inner-calibration порога, особливо LR, не доведена й не виправляється за TEST.
- Corrected прибирає додатний prior і обмежує score coverage, але не забезпечує глобальної монотонності при приховуванні суперечливих ознак і не доводить переваги за F1.
- Інформативність є ознаковою евристикою; ручні множники не є доведеною надійністю джерел; score не є каліброваною ймовірністю.
- Немає вичерпної query-wise gold-розмітки каталогу: Recall@K/Precision@K та підтверджена end-to-end якість не виміряні. Pairwise метрики не підміняють їх.
- FPV-пілот, мітки й вебдокази не змінювалися та не брали участі в навчанні. Нової інфраструктури або моделей не додано.

Кандидат і одна конфігурація **технічно підготовлені до майбутнього одноразового pairwise оцінювання**, за умови успішної фінальної звірки frozen hashes і середовища перед запуском. Це не доводить незалежної якості й не є дозволом виконати TEST. Потрібне окреме явне завдання користувача. Точний порядок — у [final_evaluation_protocol.md](final_evaluation_protocol.md). **TEST у цьому етапі не запускався.**
