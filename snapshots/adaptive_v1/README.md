# Adaptive Evidence-Aware Similarity v1

Це побайтовий знімок чотирьох модулів, залежностей, конфігурації та протоколу другого етапу. `manifest.json` містить SHA256 кожного файла й вихідного `docs/validation/results.json`, а також вибрану тоді конфігурацію adaptive: gamma=0.5, penalty=0, prior=0.20, threshold=0.15 (повна точність — у manifest).

Версія v1 — **груповий** метод `evidence_scores`; старі raw-channel формули `scores` є окремою історичною діагностикою. Знімок не змінює поточний API проєкту й не є реалізацією запропонованої v2.

Для відтворення створіть окрему робочу папку. Скопіюйте в неї знімкові `matching.py`, `audit_wdc.py`, `run_experiment.py`, `run_validation.py` та requirements-файли. Відновіть `validation_config.json` у `docs/validation_config.json`, а початкові train/validation gzip — за шляхами `data/external/`, записаними в конфігурації. Створіть Python 3.12 venv і виконайте із кореня цієї окремої папки:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 run_validation.py --train data/external/wdcproducts80cc20rnd000un_train_small.json.gz --valid data/external/wdcproducts80cc20rnd000un_valid_small.json.gz --data-config docs/validation_config.json --out results/v1_reproduction.json
```

Це відтворення вже відомого v1, а не незалежна перевірка нової формули. Воно не потребує фінального TEST. Архівні файли не редагуйте; пропозиції v2 зберігаються окремо в `docs/method_v2_proposal.md`.
