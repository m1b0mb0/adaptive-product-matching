# Відтворюваність frozen bundle v1

## Історичне середовище

Історичний preflight і тести виконано в окремому worktree на `8dcf5dbe717a9e69f422a8ebfd1b8f25dbb557f3`. Ця ревізія містить адресні правила LF/CRLF. Для Python 3.12.10 використано середовище основного checkout із пакетами з `requirements-lock.txt`. Worktree містить дві локальні копії з основного checkout, TRAIN і frozen bundle. Шлях, записаний у frozen-конфігурації для WDC TEST, у worktree відсутній.

Створіть worktree з кореня проєкту, якщо цей шлях ще не існує.

```powershell
git worktree add --detach tmp/historical-8dcf5db 8dcf5db
if ($LASTEXITCODE -ne 0) { throw 'Historical checkout failed' }

$historicalPaths = @(
    'data/external/wdcproducts80cc20rnd000un_train_small.json.gz',
    'data/processed/final_method/frozen_models.joblib'
)
foreach ($path in $historicalPaths) {
    $destination = Join-Path 'tmp/historical-8dcf5db' $path
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath $path -Destination $destination -ErrorAction Stop
}
```

Запустіть функцію `preflight()` з історичного worktree. Вона повертає контрольні суми й перевіряє модель на TRAIN. Наведена команда порівнює всі повернені поля з оригінальною квитанцією та перевіряє збережені файли за маніфестами.

```powershell
$historicalPython = (Resolve-Path '.venv/Scripts/python.exe').Path
Push-Location 'tmp/historical-8dcf5db'
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
