# Відтворюваність frozen bundle v1

## Прикладне завантаження

CLI `adaptive-corrected` використовує `frozen_v1_runtime.load_corrected_scorer()`. Для офіційної конфігурації loader перевіряє точний SHA-256 її байтів, Python 3.12.10, NumPy, pandas, SciPy, scikit-learn і joblib за зафіксованими версіями. LF-нормалізовані хеші перевіряються для джерел імпортованих `matching.py`, `adaptive_corrected.py` та `logistic_baseline.py`. SHA bundle звіряється до десеріалізації, і joblib отримує ті самі байти через потік у пам’яті. Після завантаження звіряються схема, параметри, інформативність, fitted-метадані LR, словники та IDF. Для завантаження й скорингу достатньо локальних конфігурації та bundle. Індекс назв пошуку будується окремо з каталогу.

- Типовий шлях — `docs/final_method/frozen_configuration.json`. `--method-config` також приймає побайтову копію офіційної конфігурації в межах проєкту. Помилка SHA на типовому шляху або посилання зміненої конфігурації на офіційний bundle завершує завантаження з помилкою.
- Явно передана довірена локальна конфігурація за іншим шляхом може використовувати власний bundle. Строгий режим звіряє всі хеші коду, записані в такій конфігурації, і десеріалізує перевірені байти її bundle. Якщо вона посилається на стару версію прикладного коду, використовуйте відповідний checkout. Цей режим призначений для довірених локальних joblib-файлів із відомим походженням.

## Історичне середовище

Історичний preflight і тести виконано в окремому worktree на `8dcf5dbe717a9e69f422a8ebfd1b8f25dbb557f3`. Ця ревізія містить адресні правила LF/CRLF. Для Python 3.12.10 використано середовище основного checkout із пакетами з `requirements-lock.txt`. Worktree містить дві локальні копії з основного checkout, TRAIN і frozen bundle. Шлях, записаний у frozen-конфігурації для WDC TEST, у worktree відсутній.

Створіть worktree з кореня проєкту, якщо цей шлях ще не існує. Крапка на початку імені каталогу запобігає повторному збору історичних тестів під час звичайного запуску `pytest` з кореня основного checkout.

```powershell
git worktree add --detach tmp/.historical-8dcf5db 8dcf5db
if ($LASTEXITCODE -ne 0) { throw 'Historical checkout failed' }

$historicalPaths = @(
    'data/external/wdcproducts80cc20rnd000un_train_small.json.gz',
    'data/processed/final_method/frozen_models.joblib'
)
foreach ($path in $historicalPaths) {
    $destination = Join-Path 'tmp/.historical-8dcf5db' $path
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath $path -Destination $destination -ErrorAction Stop
}
```

Запустіть функцію `preflight()` з історичного worktree. Вона повертає контрольні суми й перевіряє модель на TRAIN. Наведена команда порівнює всі повернені поля з оригінальною квитанцією та перевіряє збережені файли за маніфестами.

```powershell
$historicalPython = (Resolve-Path '.venv/Scripts/python.exe').Path
Push-Location 'tmp/.historical-8dcf5db'
try {
    @'
import json
from pathlib import Path

import final_evaluation as evaluation
from frozen_method_artifact import load_corrected_scorer

out = Path("docs/final_evaluation")
receipt = json.loads((out / "pretest_validation.json").read_bytes())
manifest = json.loads((out / "artifact_manifest.json").read_bytes())
for name, expected in manifest.items():
    assert evaluation.digest(out / name) == expected, name

snapshot = Path("snapshots/adaptive_v1")
v1 = json.loads((snapshot / "manifest.json").read_bytes())
for name, reference in v1["files"].items():
    assert evaluation.digest(snapshot / name) == reference["sha256"], name
assert evaluation.digest(Path(v1["source_results"])) == v1["source_results_sha256"]

config, _, train, actual = evaluation.preflight()
for key, value in actual.items():
    assert receipt[key] == value, key
assert actual["test_opened"] is False
assert not Path(config["test_evaluation"]["path"]).exists()
with evaluation.no_fit():
    scorer = load_corrected_scorer(evaluation.CONFIG)
    assert len(scorer.score_pairs(train[:5])) == 5
print(actual["status"], actual["test_opened"], len(train))
'@ | & $historicalPython -B -X utf8 -
    if ($LASTEXITCODE -ne 0) { throw 'Historical verification failed' }
    & $historicalPython -B -m pytest -q -p no:cacheprovider tests/test_final_evaluation.py tests/test_final_method_pipeline.py tests/test_v1_snapshot.py
    if ($LASTEXITCODE -ne 0) { throw 'Historical tests failed' }
} finally {
    Pop-Location
}
```

Фактичний результат preflight — `PASSED`, `test_opened=False`, 2 500 пар TRAIN. Історичний loader також завантажив модель і повернув оцінки для контрольної підмножини TRAIN. Збіглися 11 LF-нормалізованих хешів коду, три побайтові хеші коду з квитанції, 23 хеші фінальних артефактів і дев’ять хешів snapshot v1 та пов’язаного результату. Оригінальні історичні тести — **22 пройшли**.

| Файл | Фактичний SHA-256 |
|---|---|
| Frozen-конфігурація | `2976e18752a2463a7ef784714d5468a150db0a9e0142812d471e3fc13b0c655a` |
| WDC TRAIN | `1915a92de76ddbf63a6c4d7ff3162df98f505033a8ad12ee7072b78b083d77c6` |
| Frozen bundle | `329fa0919f570172956b3fe60aa97dce68357f8189370ea317f3221487ff5caa` |

## Числовий контракт застосунку

Сім синтетичних пар перевіряють відсутні поля, порожній текст і пробіли, текст поза словником (OOV), нульову подібність брендів, відсутній бренд і повні записи зі збігом та конфліктом. Еталони отримано історичним `FrozenCorrectedScorer` на оригінальному bundle з увімкненою забороною fit. Тести звіряють порядок `title, description, brand, identifier`, точну маску доступності й значення з абсолютним допуском `1e-14`. Окремі перевірки звіряють хеші TRAIN-словників та IDF і результати `SearchEngine` на мінікаталозі з метаданими. Побудова окремого індексу назв каталогу дозволена під час створення рушія. [Еталони й тести](../../tests/test_frozen_v1_contract.py)

Звірка чисел безпосередньо в історичному worktree, запуск з кореня основного checkout.

```powershell
$historicalPython = (Resolve-Path '.venv/Scripts/python.exe').Path
$frozenContract = (Resolve-Path 'tests/test_frozen_v1_contract.py').Path
Push-Location 'tmp/.historical-8dcf5db'
try {
    @'
import runpy
import sys
from pathlib import Path

import numpy as np
import final_evaluation as evaluation
from frozen_method_artifact import load_corrected_scorer

assert Path(evaluation.__file__).resolve().parent == Path.cwd()
contract = runpy.run_path(sys.argv[1])
cases = contract["CASES"]
with evaluation.no_fit():
    scorer = load_corrected_scorer(evaluation.CONFIG)
    matrix = scorer.extractor.transform(list(cases.values())).matrix
    scores = scorer.score_pairs(list(cases.values()))
for case, features, score in zip(cases, matrix, scores, strict=True):
    expected_features, expected_score = contract["REFERENCE"][case]
    np.testing.assert_allclose(features, np.array(expected_features, dtype=float), rtol=0, atol=1e-14)
    np.testing.assert_allclose(score, expected_score, rtol=0, atol=1e-14)
print(f"{len(cases)} historical numerical references match")
'@ | & $historicalPython -B -X utf8 - $frozenContract
    if ($LASTEXITCODE -ne 0) { throw 'Historical scoring comparison failed' }
} finally {
    Pop-Location
}
```

У поточному checkout повний `pytest` запускає `tests/test_historical_evaluation.py`. Він матеріалізує ревізію `8dcf5db` у тимчасовому каталозі через окремий Git-індекс і копіює тільки TRAIN і bundle. Дочірній pytest виконує оригінальні історичні тести. Зовнішній тест вимагає успішний код виходу, рівно 22 результати в JUnit XML і відсутність пропусків, помилок або провалів. Помилки checkout чи відсутні локальні артефакти також провалюють тест. Поточні прикладні перевірки виконуються в основному checkout.

Команди перевірки поточного checkout.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q --tb=short tests/test_frozen_v1_runtime.py tests/test_frozen_v1_contract.py tests/test_corrected_search.py tests/test_search_cli.py
.\.venv\Scripts\python.exe -B -m pytest -q --tb=short
.\.venv\Scripts\python.exe -m pip check
```

Перевірка демонстрацій додатково потребує локального FPV offers CSV, тому її запускають окремо. Вона звіряє повні JSON-об’єкти для п’яти збережених демонстрацій. Демонстрація corrected перевіряється двічі, з типовим і явним шляхом конфігурації. [Маніфест демонстрацій](demo_manifest.json)

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/check_search_demos.py
```
