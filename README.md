# Інтелектуальна система адаптивного пошуку товарних пропозицій

Python-прототип програмної частини магістерської роботи. Реалізовано аудит WDC Products і попередні **validation-only** порівняння методів. Фінальне оцінювання на зарезервованому test не виконувалося. Метрики стосуються класифікації готових пар, а не повноти вебпошуку чи генерації кандидатів.

- [Методика й формули](docs/experiment_methodology.md).
- [Валідаційні результати, абляції та графіки](docs/validation_results.md).
- [JSON результатів](docs/validation/results.json), [CSV метрик](docs/validation/metrics.csv), [зафіксований протокол](docs/validation/protocol.json).
- [Початковий аудит](docs/data_audit.md), [перевірка valpair](docs/validation_data_audit.json), [конфігурація validation](docs/validation_config.json).

## Середовище

Перевірено на Windows, CPython **3.12.10**, ізольованому `.venv` без системних пакетів. PowerShell із кореня проєкту:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`python` має вказувати на Python 3.12; якщо доступний Python Launcher, можна використати `py -3.12 -m venv .venv`. Активація середовища не потрібна. `requirements-lock.txt` містить перевірені точні версії, `requirements.txt` — дозволені діапазони. Тестові фікстури перевіряють логіку й не є джерелом наукових результатів; для pytest WDC не потрібний.

## Дані й аудит

Завантажте [офіційний 80pair.zip](https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip) зі [сторінки WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) у `data/external/80pair.zip`. Сирі дані в GitHub не зберігаються.

```powershell
.\.venv\Scripts\python.exe list_archive.py data/external/80pair.zip
.\.venv\Scripts\python.exe -X utf8 audit_wdc.py --archive data/external/80pair.zip --output-dir docs --extract-dir data/external
.\.venv\Scripts\python.exe -X utf8 prepare_validation.py --download
```

Перший аудит перевіряє всі 9 gzip JSONL-файлів і розпаковує лише small train/validation та зарезервований 100un test. Оригінальний ZIP не змінюється. Його SHA256: `b2044939cee5ea6f12148a2f3551508de3cb77660dfc91767c44daaf9d8a9c4a`.

`prepare_validation.py --download` завантажує додатковий офіційний `val_pair.zip` лише за відсутності локального `data/external/valpair.zip`. Якщо архів уже є, `--download` не потрібний. Перевіряються вкладена структура, хеші та витік даних; різні наявні архіви/файли не перезаписуються. Зарезервований test читається лише через metadata-проєкцію для порівняння ID, кластерів і текстових відбитків; його мітки й метрики не аналізуються.

**Фактична проблема valpair:** додаткові small 050un/100un мають однаковий JSONL і містять 110 train-пропозицій та 27 train-пар. Вони відхилені. Використано дозволений fallback — початкову seen-validation без зміни рядків. Конфігурація першого аудиту `docs/wdc_config.json` збережена як історичний артефакт; поточний вибір — `docs/validation_config.json`.

| Роль | Файл у `data/external/` | Пари | Позитивні / негативні |
| --- | --- | ---: | ---: |
| Train | `wdcproducts80cc20rnd000un_train_small.json.gz` | 2500 | 500 / 2000 |
| Validation, seen fallback | `wdcproducts80cc20rnd000un_valid_small.json.gz` | 2500 | 500 / 2000 |
| Зарезервований test | `wdcproducts80cc20rnd100un_gs.json.gz` | — | У цьому етапі не аналізуються |

80% corner cases — параметр добору складних кластерів, а не частка hard-negative пар. Спільні train/validation кластери відповідають seen-режиму, тоді як повторних offer IDs між ними немає. Перенесення результатів на unseen-test не доведено.

## Відтворення валідаційного порівняння

```powershell
.\.venv\Scripts\python.exe -X utf8 run_validation.py --train data/external/wdcproducts80cc20rnd000un_train_small.json.gz --valid data/external/wdcproducts80cc20rnd000un_valid_small.json.gz --data-config docs/validation_config.json --out docs/validation/results.json
.\.venv\Scripts\python.exe -X utf8 report_validation.py --results docs/validation/results.json --out-dir docs/validation --report docs/validation_results.md
```

`run_validation.py` приймає лише train і validation; аргументу `--test` немає. Перед fit записується `protocol.json`. У JSON результатів є хеші коду/даних і вибрані на tuning параметри. Друга команда створює Markdown, CSV, PNG/SVG з готового JSON без навчання або добору параметрів.

TF-IDF та field discriminativeness оцінюються лише на train. Validation детерміновано розділено за кластерами, seed `20261101`: TUNE — 729 пар (250 позитивних), REPORT — 728 (250 позитивних). Між частинами немає спільних кластерів чи пропозицій; 1043 негативні пари між частинами виключено з обох, вихідний файл збережено. Такий внутрішній протокол змінює баланс класів і не відтворює опубліковану повну benchmark-оцінку.

Порівнюються title TF-IDF/cosine, fixed weighted, available-group renormalization та adaptive evidence-aware similarity. Три атрибутні методи мають однакові ознаки й групи `{title, identifier}`, `{description}`, `{brand}` із бюджетами 0.70/0.15/0.15. Усередині першої групи збережено співвідношення 4/7 і 3/7. Модельні токени з назви більше не збільшують coverage як незалежне джерело.

Adaptive додає train-розрізнювальність, ручні множники надійності та shrinkage за покриттям груп. Main penalty=0, prior reference=0.20; gamma із 0.5/1.0/1.5 і пороги 0.01–0.99 обираються лише на TUNE. Окремо перевіряються абляції coverage/informativeness/reliability із власним TUNE-порогом та порогом повної моделі. Історичні raw-attribute методи з префіксом `legacy_` — додаткова діагностика.

На REPORT маскуються лише description/brand: 0/10/20/30/50%, seeds 11/22/33, одна маска на offer ID/поле для всіх пар і методів. Пороги та параметри після маскування не змінюються. Графіки показують середнє ± вибіркове SD масок, не довірчий інтервал. Фактична частка втрати порівнюваних полів пари теж збережена.

## Межі інтерпретації

Ідентифікатор — евристика з назви й залишається корельованим із нею. «Інформативність» тут означає глобальну розрізнювальність каналу, а не інформацію окремого значення. Надійність `[1,1,1,0.70]` задано вручну: це гіпотеза для sensitivity-абляції, а не доведена точність вилучення чи надійність вебджерела. Coverage і score не є каліброваними ймовірностями. WDC не містить окремих KV/статора/напруги FPV-двигунів. Наукову новизну й перевагу adaptive ще не доведено.

`matching.scores`, `run_experiment.py`, `plot_results.py` та `search_cli.py` збережені для відтворення стартового прототипу. Поточні grouped-методи використовують `matching.evidence_scores`, `run_validation.py` та `report_validation.py`; старий фінальний runner на цьому етапі не запускається. Пошукова CLI-демонстрація навчає TF-IDF на колекції й не дає незалежних test-метрик.

`.gitignore` виключає raw/external/processed дані, архіви, `.venv`, кеші, `tmp/` і великі результати. Компактні validation JSON/CSV/PNG/SVG у `docs/validation/` дозволені для збереження разом із кодом. `docs/code_validation.json` фіксує історичний стан першого аудиту; актуальні перевірки другого етапу — `docs/validation/pipeline_validation.json`. Шаблон `data/motor_offers_template.csv` і правила анотування — заготовки окремого майбутнього збору даних.
