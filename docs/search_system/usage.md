# Запуск пошукового прототипу

Потік: імпорт → нормалізація → TF-IDF-кандидати → similarity → ранжування → консоль/JSON. [Компоненти й схема даних](architecture.md), [розширення для frozen LR](logistic_extension.md).

Потрібне наявне середовище Python 3.12 із залежностями проєкту; нових пакетів немає. Команди виконуються з кореня проєкту в PowerShell.

## WDC TRAIN

```powershell
.\.venv\Scripts\python.exe -X utf8 search_cli.py --catalog-format wdc --catalog data/external/wdcproducts80cc20rnd000un_train_small.json.gz --query-id 1017836 --candidate-k 20 --top 5
```

Імпортуються 1000 унікальних пропозицій. Запит `1017836` — `APC Smart-UPS SRT 6000VA RM 230V`: для демонстрації взято лексикографічно найменший ID із непорожньою назвою, без добору за якістю видачі. CLI звіряє назву TRAIN-файлу й SHA-256; інші splits не приймаються.

У WDC TRAIN немає магазинів і URL. Вони залишаються невідомими; додавання `--other-stores-only` дасть зрозумілу помилку. Назва датасету, бренд і offer ID не підміняють магазин.

Сумісний старий аргумент: `--pairs data/external/wdcproducts80cc20rnd000un_train_small.json.gz` замість `--catalog`. Він підтримує тільки WDC TRAIN.

## FPV-пілот та інші магазини

```powershell
.\.venv\Scripts\python.exe -X utf8 search_cli.py --catalog-format fpv --catalog data/external/fpv_motors/fpv_motors_offers_pilot.csv --query-id FP04 --candidate-k 20 --top 5 --other-stores-only
```

Імпортуються лише 32 вихідні пропозиції з 6 магазинів. Candidate CSV, review queue та їхні мітки не потрібні й не читаються. FP04 — пропозиція FPVUA `Мотор RCINPOWER Wasp Major V2 Champion Edition 22.6-6.6 2160KV`. Фільтр виключає FPVUA й невідомі магазини до відбору top-K.

Для Fixed Weighted додайте `--method fixed`; default — відомий baseline Renormalization, а не авторський метод:

```powershell
.\.venv\Scripts\python.exe -X utf8 search_cli.py --catalog-format fpv --catalog data/external/fpv_motors/fpv_motors_offers_pilot.csv --query-id FP04 --candidate-k 20 --top 5 --other-stores-only --method fixed
```

Технічні поля зберігаються як спостережувані метадані. Наприклад, `voltage_s` у DR01 залишається порожнім, хоча запит FP04 має це поле. Неоднозначні DR05/DR06 залишаються каталоговими записами зі своїми вихідними статусами.

## Параметри та помилки

| Аргумент | Призначення |
| --- | --- |
| `--catalog` | Шлях до WDC TRAIN gzip JSONL або FPV offers CSV |
| `--catalog-format wdc\|fpv` | Формат; default `wdc`, також `--format` |
| `--query-id` | ID конкретної пропозиції з каталогу |
| `--candidate-k` | Максимум TF-IDF-кандидатів, default 20 |
| `--top` | Максимум результатів після ранжування, default 5; також `--top-k` |
| `--method fixed\|renorm` | Метод ранжування, default `renorm` |
| `--other-stores-only` | Лише відомі інші магазини; також `--other-sources-only` |
| `--json` | Машинний JSON замість консольного представлення |
| `--pairs` | Старий аргумент тільки для WDC TRAIN, взаємовиключний із `--catalog` |

Ліміти — додатні цілі числа. Якщо `top > candidate_k` або придатних кандидатів менше, результатів буде менше. Ліміти 20/5 обрано для демонстрації, без оптимізації за мітками чи development outer-результатами.

Порожній каталог, невідомий ID, пошкоджений файл, конфліктний повтор ID або невідомий магазин запиту при ввімкненому фільтрі дають повідомлення й exit code 2. Порожня/непредставна назва або відсутність ненульового текстового перетину дають порожню видачу. Випадкові товари для заповнення top-K не додаються.

Індекс і pairwise-представлення будуються на каталозі без міток при створенні рушія. Повторні запити до того самого об'єкта не викликають `fit`; окремі запуски CLI створюють індекс заново:

```python
from product_catalog import load_fpv_catalog
from search_system import SearchEngine

catalog = load_fpv_catalog("data/external/fpv_motors/fpv_motors_offers_pilot.csv")
engine = SearchEngine(catalog, method="renorm")
candidates = engine.retrieve("FP04", candidate_k=20, other_sources_only=True)
results = engine.search("FP04", candidate_k=20, top_k=5, other_sources_only=True)
```

## Збережені реальні демонстрації

| Каталог / метод | Пропозицій | Кандидатів | Результатів | ID найвищого score |
| --- | ---: | ---: | ---: | --- |
| WDC / Renormalization | 1000 | 20 | 5 | 84668470 |
| WDC / Fixed | 1000 | 20 | 5 | 84668470 |
| FPV / Renormalization | 32 | 20 | 5 | DR01 |
| FPV / Fixed | 32 | 20 | 5 | DR01 |

- [WDC Renormalization, консоль](wdc_renorm_demo.txt) / [JSON](wdc_renorm_demo.json), [WDC Fixed](wdc_fixed_demo.txt) / [JSON](wdc_fixed_demo.json).
- [FPV Renormalization, консоль](fpv_renorm_demo.txt) / [JSON](fpv_renorm_demo.json), [FPV Fixed](fpv_fixed_demo.txt) / [JSON](fpv_fixed_demo.json).
- [Маніфест](demo_manifest.json): точні команди, хеші двох джерел, правила вибору запитів.
- [Перевірка pipeline](pipeline_validation.json): тести, звіряння демонстрацій і незмінності історичних файлів.

Для FP04 найвище ранжується DR01 із Drono Store: `similarity_score=0.927338`, `retrieval_score=0.911767` у Renormalization. Це оцінки алгоритмів, не ймовірність або підтверджена правильна відповідність. URL та назва походять із CSV; вебсторінка під час пошуку не перевіряється.

Різницю етапів видно на WDC: 61195551 має retrieval score приблизно 0.721782, а 84668470 — 0.503881; після багатоканального ранжування 84668470 має вищий similarity score і стає першим результатом. Retrieval використовує лише назви; matching оцінює вже відібрані пари за чотирма наявними similarities. Правильність нового порядку без еталонної перевірки не встановлюється.

## Автоматизовані перевірки

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Перевірено на Python 3.12.10: **432 тести пройшли**, з них 103 нові перевірки імпорту, retrieval, ранжування, frozen LR та CLI. `pip check` не виявив порушень залежностей. Усі 96 зафіксованих попередніх файлів коду, тестів, звітів, snapshots і FPV-даних збережено побайтово. Свідомі зміни наявних файлів — `search_cli.py` і README. Деталі та хеші — у `pipeline_validation.json`.

## Реалізовані можливості й обмеження

Готові єдина модель пропозиції, два імпортери, label-free TF-IDF retrieval, детерміновані Fixed/Renormalization, фільтр магазину, CLI/JSON, тести та in-memory точка підключення frozen LR. Технічні FPV-параметри відображаються, але не зіставляються новою формулою.

Не реалізовані серіалізація/CLI-завантаження погодженого LR-артефакта, structured FPV matcher, довільний запит поза каталогом, онлайн-збирання сторінок чи вебінтерфейс. Вони не потрібні для поточного локального сценарію.

Recall@K та інші метрики якості пошуку не обчислювалися: парні мітки не є вичерпною розміткою всіх релевантних результатів каталогу. Пропуски можуть обмежити retrieval; подальше ранжування не повертає кандидатів, які не пройшли відбір. FPV-мітки не підтверджені. Історичні експерименти й дані збережено; REPORT і зарезервований TEST для цього етапу не використовувалися. Нові параметри та Adaptive v2/v3 не розроблялися.
