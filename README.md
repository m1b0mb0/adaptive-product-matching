# Інтелектуальна система адаптивного пошуку товарних пропозицій

Дослідницький Python-прототип для магістерської роботи. Поточний етап — **аудит коду й реального WDC Products**, завершений 1 листопада 2026 року. Порівняння методів на benchmark ще не виконувалося; наукових результатів Precision/Recall/F1 поки немає. Програма класифікує готові пари пропозицій.

Звіт: [docs/data_audit.md](docs/data_audit.md). Машинозчитувані матеріали: [data_audit.json](docs/data_audit.json), [archive_manifest.json](docs/archive_manifest.json), [wdc_config.json](docs/wdc_config.json). Дані не потрібно зберігати в GitHub.

## Середовище й тести

Перевірено на Windows, CPython **3.12.10**, в ізольованому `.venv` без системних пакетів. Команди PowerShell із кореня проєкту:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`python` має вказувати на Python 3.12; за наявності Windows Python Launcher можна створити середовище командою `py -3.12 -m venv .venv`. Активація середовища не потрібна. `requirements-lock.txt` фіксує перевірені версії залежностей; `requirements.txt` містить дозволені діапазони для майбутнього оновлення. Тести використовують малі контрольні приклади для перевірки логіки, а не для отримання наукових результатів; датасет їм не потрібен.

## Датасет і відтворення аудиту

Завантажте **80pair.zip** зі [сторінки WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) або за [прямим офіційним посиланням](https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip) та покладіть у `data/external/80pair.zip`. У перевіреній папці він уже був наявний. SHA256 локального архіву:

```text
b2044939cee5ea6f12148a2f3551508de3cb77660dfc91767c44daaf9d8a9c4a
```

```powershell
.\.venv\Scripts\python.exe list_archive.py data/external/80pair.zip
.\.venv\Scripts\python.exe -X utf8 audit_wdc.py --archive data/external/80pair.zip --output-dir docs --extract-dir data/external
```

Команда перевіряє всі 9 gzip JSONL-файлів, фіксує контрольні суми, структуру й статистики та розпаковує лише три вибрані файли. Оригінальний ZIP не змінюється; наявні файли з іншою контрольною сумою не перезаписуються. Без `--extract-dir` аудит читає дані безпосередньо з архіву. Якщо зберігати архів у `data/raw/`, передайте відповідний шлях у `--archive`.

Зафіксована конфігурація: **80% corner-case clusters, small, 100% unseen test**. Це параметр формування складних товарних кластерів, а не 80% складних негативних пар.

| Роль | Файл у `data/external/` | Пари | Позитивні / негативні |
| --- | --- | ---: | ---: |
| Train | `wdcproducts80cc20rnd000un_train_small.json.gz` | 2500 | 500 / 2000 |
| Validation | `wdcproducts80cc20rnd000un_valid_small.json.gz` | 2500 | 500 / 2000 |
| Test | `wdcproducts80cc20rnd100un_gs.json.gz` | 4500 | 500 / 4000 |

`gs` — тестовий gold standard. Train і validation з `000un` є спільними для офіційних seen/unseen тестових режимів. Test `100un` має інші товарні кластери; перевірка не виявила спільних offer IDs або пар із train/validation. Validation залишається seen-model, що створює відмінність розподілів із test. Окремий додатковий unseen-validation архів не входить у `80pair.zip` і під час цього аудиту не завантажувався. В альтернативному `000un_gs` знайдено 4 спільні offer IDs з `valid_small`; `050un_gs` теж має перетини пропозицій і пар із small train/validation. Дані не змінювалися. Подробиці — у звіті.

Можна також перевірити будь-які три розпаковані JSONL/gzip JSONL-файли:

```powershell
.\.venv\Scripts\python.exe audit_wdc.py --train data/external/wdcproducts80cc20rnd000un_train_small.json.gz --valid data/external/wdcproducts80cc20rnd000un_valid_small.json.gz --test data/external/wdcproducts80cc20rnd100un_gs.json.gz --out docs/data_audit_explicit.json
```

## Що реалізовано

- `audit_wdc.py`: суворе читання кожного рядка, пропуски на рівні сторін пар та унікальних пропозицій, класи, дублікати, конфлікти, offer/pair/cluster/text overlap, архів і конфігурація.
- `matching.py`: NFKC/casefold/пробіли, TF-IDF назв/описів, точне зіставлення бренду, евристичні модельні токени з назви; fixed/renorm/adaptive scores і маскування.
- `run_experiment.py`: навчання представлень і ваг на train, вибір порогів та параметрів на validation, метрики на test і три абляції. За наявності порушень цілісності чи спільних offer IDs виконання зупиняється.
- `search_cli.py`: демонстрація схожих пропозицій; TF-IDF навчається на колекції демонстрації, тому її scores не є незалежними test-метриками.
- `plot_results.py`: графік F1 проти додаткових пропусків після майбутнього експерименту.

Відсутні значення (`None`, числовий NaN, pandas NA/NaT, порожні рядки) не утворюють тексту `nan`. Некоректні JSON-типи reader відхиляє. Непорівнювані поля позначаються `NaN` у матриці ознак; це також стосується TF-IDF із нульовим вектором поза навчальним словником. Збіг двох відсутніх значень не дає позитивних доказів. Ціна, валюта, ID, `cluster_id` і службові прапорці не є ознаками тотожності.

Адаптивний кандидат поки використовує розрізнювальність **поля** на train, фіксовану довіру до евристичних ідентифікаторів, покриття ознак і штраф за їхній конфлікт. Це попередня реалізація: вона не оцінює надійність вебджерел і не реалізує інформативність кожного окремого значення атрибута. Ідентифікатор походить із назви й не є незалежним структурованим полем. WDC дає реальні пропуски описів/брендів, але не окремі KV, розміри статора чи інші FPV-параметри.

## Команди для наступного етапу

Наведені для відтворюваності; під час аудиту порівняльний запуск **не виконувався**. До нього слід зафіксувати протокол і обмеження зі звіту. Test не використовуйте для підбору параметрів або вибору seen/unseen режиму за отриманим F1.

```powershell
New-Item -ItemType Directory -Force results | Out-Null
.\.venv\Scripts\python.exe run_experiment.py --train data/external/wdcproducts80cc20rnd000un_train_small.json.gz --valid data/external/wdcproducts80cc20rnd000un_valid_small.json.gz --test data/external/wdcproducts80cc20rnd100un_gs.json.gz --out results/wdc_small_100un.json
.\.venv\Scripts\python.exe plot_results.py --results results/wdc_small_100un.json --out results/f1_missingness.png
```

Метрики — Precision, Recall, F1, TP/FP/FN/TN на заданих парах; вони не оцінюють повний пошук або recall генерації кандидатів. Маскування стосується лише brand/description на рівні пропозиції: 0/10/20/30/50%, seeds 11/22/33, пороги після маскування не змінюються. Частка маскованих пар може бути вищою за частку маскованих пропозицій. Затінення графіка — стандартне відхилення між трьома масками, не довірчий інтервал.

`.gitignore` виключає raw/external/processed дані, архіви, середовища, кеші, `results/` та тимчасові артефакти. Невеликі звіти `docs/*.json` і документація залишаються придатними для збереження. Шаблон `data/motor_offers_template.csv` та `data/annotation_guidelines.md` збережено: це заготовки майбутнього окремого збору, а не отриманий датасет.
