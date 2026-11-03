# Аналіз помилок на TRAIN development

Нижче аналізуються outer-held-out пари з TRAIN. Мітки FP/FN є помилками відносно WDC gold; зміст товарних сторінок повторно не перевірявся. Автоматичні теги описують ознаки, можуть перетинатися і не доводять причину помилки.

| Метод | Помилка | Діагностичний тег | Кількість |
| --- | --- | --- | --- |
| TF-IDF + cosine | FN | ALL | 6 |
| TF-IDF + cosine | FN | identifier_zero_overlap_proxy | 1 |
| TF-IDF + cosine | FN | some_unavailable_channels | 6 |
| TF-IDF + cosine | FP | ALL | 253 |
| TF-IDF + cosine | FP | brand_unequal_strings | 10 |
| TF-IDF + cosine | FP | high_title_similarity | 16 |
| TF-IDF + cosine | FP | identifier_zero_overlap_proxy | 88 |
| TF-IDF + cosine | FP | other_lexical_case | 4 |
| TF-IDF + cosine | FP | some_unavailable_channels | 238 |
| Fixed weighted | FN | ALL | 10 |
| Fixed weighted | FN | identifier_zero_overlap_proxy | 6 |
| Fixed weighted | FN | some_unavailable_channels | 10 |
| Fixed weighted | FP | ALL | 245 |
| Fixed weighted | FP | brand_unequal_strings | 9 |
| Fixed weighted | FP | high_title_similarity | 16 |
| Fixed weighted | FP | identifier_zero_overlap_proxy | 79 |
| Fixed weighted | FP | other_lexical_case | 4 |
| Fixed weighted | FP | some_unavailable_channels | 230 |
| Renormalization | FN | ALL | 10 |
| Renormalization | FN | identifier_zero_overlap_proxy | 5 |
| Renormalization | FN | some_unavailable_channels | 10 |
| Renormalization | FP | ALL | 249 |
| Renormalization | FP | brand_unequal_strings | 9 |
| Renormalization | FP | high_title_similarity | 16 |
| Renormalization | FP | identifier_zero_overlap_proxy | 83 |
| Renormalization | FP | other_lexical_case | 4 |
| Renormalization | FP | some_unavailable_channels | 234 |
| Adaptive v1 | FN | ALL | 10 |
| Adaptive v1 | FN | identifier_zero_overlap_proxy | 5 |
| Adaptive v1 | FN | some_unavailable_channels | 10 |
| Adaptive v1 | FP | ALL | 247 |
| Adaptive v1 | FP | brand_unequal_strings | 7 |
| Adaptive v1 | FP | high_title_similarity | 16 |
| Adaptive v1 | FP | identifier_zero_overlap_proxy | 82 |
| Adaptive v1 | FP | other_lexical_case | 4 |
| Adaptive v1 | FP | some_unavailable_channels | 232 |
| Logistic Regression | FN | ALL | 53 |
| Logistic Regression | FN | brand_unequal_strings | 1 |
| Logistic Regression | FN | identifier_zero_overlap_proxy | 21 |
| Logistic Regression | FN | other_lexical_case | 1 |
| Logistic Regression | FN | some_unavailable_channels | 52 |
| Logistic Regression | FP | ALL | 199 |
| Logistic Regression | FP | brand_unequal_strings | 6 |
| Logistic Regression | FP | high_title_similarity | 16 |
| Logistic Regression | FP | identifier_zero_overlap_proxy | 56 |
| Logistic Regression | FP | other_lexical_case | 4 |
| Logistic Regression | FP | some_unavailable_channels | 186 |

Повний перелік — `errors.csv`; усі рішення, включно з правильними, — `clean_predictions.csv`. Оцінювання hard negatives наведено в `results.md`; офіційний hard-прапорець не визначається за score моделі.

## Приклади FP та FN

Наведено детерміновано перші дві помилки кожного типу для кожного методу за pair ID, без відбору найпереконливіших прикладів. Назви — оригінальні значення; довгі назви скорочено для читання.

### TF-IDF + cosine

| Тип | Pair ID | Hard | Ліва назва | Права назва | Score | Поріг | Теги |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 10999896#3762458 | True | DVR Hikvision TurboHD 4 canale DS-7204HQHI-K1 | DS-7216HQHI-K1 Hikvision 16 cs. TurboHD DVR | 0.7014 | 0.11 | identifier_zero_overlap_proxy;some_unavailable_channels |
| FP | 1136383#41520329 | True | Evans EC2 Clear Drum Head, 14 Inch | Evans EC2 SST Clear Batter Drumhead 13 in. | 0.5979 | 0.13 | some_unavailable_channels |
| FN | 12432797#32885383 | False | Canon 24-70mm f/2.8L II USM EF Lens | Nikon AF-S NIKKOR FX 50mm F1.8G Lens | 0.1373 | 0.19 | some_unavailable_channels |
| FN | 18456007#94106554 | False | RAM® Composite Double Socket Arm | RAM Mounts short arm for 1\" ball | 0.1086 | 0.13 | some_unavailable_channels |

### Fixed weighted

| Тип | Pair ID | Hard | Ліва назва | Права назва | Score | Поріг | Теги |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 10999896#3762458 | True | DVR Hikvision TurboHD 4 canale DS-7204HQHI-K1 | DS-7216HQHI-K1 Hikvision 16 cs. TurboHD DVR | 0.2806 | 0.09 | identifier_zero_overlap_proxy;some_unavailable_channels |
| FP | 1136383#41520329 | True | Evans EC2 Clear Drum Head, 14 Inch | Evans EC2 SST Clear Batter Drumhead 13 in. | 0.4600 | 0.11 | some_unavailable_channels |
| FN | 12432797#32885383 | False | Canon 24-70mm f/2.8L II USM EF Lens | Nikon AF-S NIKKOR FX 50mm F1.8G Lens | 0.0961 | 0.14 | some_unavailable_channels |
| FN | 33609155#93094504 | False | Maintenance Unit Epson T6190 35000Pgs (C13T619000) | maintenance kit EPSON Business Inkjet B300/310/B500DN/B510DN (´odpad. nadoba´) | 0.1179 | 0.14 | identifier_zero_overlap_proxy;some_unavailable_channels |

### Renormalization

| Тип | Pair ID | Hard | Ліва назва | Права назва | Score | Поріг | Теги |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 10999896#3762458 | True | DVR Hikvision TurboHD 4 canale DS-7204HQHI-K1 | DS-7216HQHI-K1 Hikvision 16 cs. TurboHD DVR | 0.4008 | 0.11 | identifier_zero_overlap_proxy;some_unavailable_channels |
| FP | 1136383#41520329 | True | Evans EC2 Clear Drum Head, 14 Inch | Evans EC2 SST Clear Batter Drumhead 13 in. | 0.5412 | 0.19 | some_unavailable_channels |
| FN | 18456007#94106554 | False | RAM® Composite Double Socket Arm | RAM Mounts short arm for 1\" ball | 0.1774 | 0.19 | some_unavailable_channels |
| FN | 35469349#19949293 | False | WD Blue 3D - 2048 GB - M.2 - 560 MB/s - 6 Gbit/s WDS200T2B0B | SSD Western Digital Blue 2TB M.2 2280 TLC SATA SSD | 0.1112 | 0.13 | some_unavailable_channels |

### Adaptive v1

| Тип | Pair ID | Hard | Ліва назва | Права назва | Score | Поріг | Теги |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 10999896#3762458 | True | DVR Hikvision TurboHD 4 canale DS-7204HQHI-K1 | DS-7216HQHI-K1 Hikvision 16 cs. TurboHD DVR | 0.3430 | 0.13 | identifier_zero_overlap_proxy;some_unavailable_channels |
| FP | 1136383#41520329 | True | Evans EC2 Clear Drum Head, 14 Inch | Evans EC2 SST Clear Batter Drumhead 13 in. | 0.5240 | 0.19 | some_unavailable_channels |
| FN | 18456007#94106554 | False | RAM® Composite Double Socket Arm | RAM Mounts short arm for 1\" ball | 0.1677 | 0.19 | some_unavailable_channels |
| FN | 35469349#19949293 | False | WD Blue 3D - 2048 GB - M.2 - 560 MB/s - 6 Gbit/s WDS200T2B0B | SSD Western Digital Blue 2TB M.2 2280 TLC SATA SSD | 0.1257 | 0.14 | some_unavailable_channels |

### Logistic Regression

| Тип | Pair ID | Hard | Ліва назва | Права назва | Score | Поріг | Теги |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 10999896#3762458 | True | DVR Hikvision TurboHD 4 canale DS-7204HQHI-K1 | DS-7216HQHI-K1 Hikvision 16 cs. TurboHD DVR | 0.4290 | 0.26 | identifier_zero_overlap_proxy;some_unavailable_channels |
| FP | 1136383#41520329 | True | Evans EC2 Clear Drum Head, 14 Inch | Evans EC2 SST Clear Batter Drumhead 13 in. | 0.5140 | 0.20 | some_unavailable_channels |
| FN | 10719102#16785446 | False | 3M - PF27.0W9 PRIVACY FILTER BLACK FOR 27,0IN / 68,6 CM / 16:9IN ACCS | 3M - Sekretessfilter 27\" Widescreen 16:9 | 0.2916 | 0.30 | some_unavailable_channels |
| FN | 12243414#80102860 | False | Canon CLI-251BK XL High-Capacity Black Ink Tank | Canon 6448B001 (CLI-251XL) Ink Cartridge, HY Black, OEM | 0.2421 | 0.30 | identifier_zero_overlap_proxy;some_unavailable_channels |

## Що змінила Logistic Regression відносно Adaptive v1

| Перехід | Кількість |
| --- | --- |
| corrected_fp | 48 |
| introduced_fn | 43 |
| introduced_fp | 0 |
| corrected_fn | 0 |

Менша кількість загальних помилок не гарантує вищого F1: виправлення FP та поява FN змінюють баланс Precision/Recall. Це зафіксовані парні рішення, а не результат нового підбору порога.

Серед 199 FP Logistic Regression офіційний hard-прапорець мають 199. Серед 53 FN 52 мають хоча б один недоступний канал; identifier доступний у 25, із них 21 мають нульовий Jaccard; brand порівнюється лише у 5. Ці ознаки можуть співіснувати; їхня наявність не доводить причини FN.

### Ілюстрації обмежень лексичних ознак

Ці приклади обрано для пояснення різних механізмів, а не оцінювання їхньої частоти. Вони є FP відносно незмінених WDC labels; товарні сторінки не перевірялися.

| Pair ID | Ліва назва | Права назва | Identifier | Спостереження |
| --- | --- | --- | --- | --- |
| 12615558#57337098 | Logitech H570e Monaural USB Headset | LOGITECH Logitech USB Headset H390 | недоступний | Logitech H570e / H390: різні model-like позначення; identifier недоступний. Чотирисимвольний H390 не проходить незмінений мінімум довжини 5 після канонізації. |
| 47146148#46246978 | Fujifilm Instax Square Film (10 Pack) | Fujifilm Instax mini Rainbow film 10 pack | недоступний | Instax Square / Instax mini Rainbow: спільні слова не кодують усі відмінності формату товару. |
| 56034511#80086981 | TP-Link AC600 T2U Wireless USB Nano Adapter | TP-Link Archer T9UH Wireless AC1900 USB Adapter | 0.0 | TP-Link T2U / T9UH: помилка залишається навіть за спостережуваного identifier Jaccard = 0. |

### Пара для перевірки якості gold labels

Gold-positive `12432797#32885383` містить назви `Canon 24-70mm f/2.8L II USM EF Lens` та `Nikon AF-S NIKKOR FX 50mm F1.8G Lens`. Видима суперечність бренду й моделі потребує ручного дослідження джерел та кластерної анотації. Тут мітка збережена, пара не видаляється з метрик і не перерозмічається автоматично. Цей приклад застерігає від інтерпретації кожного gold-relative FN як очевидної помилки алгоритму.

## Межі інтерпретації

Однакові brand або title/model-like токени можуть належати різним модифікаціям; низька текстова подібність може приховувати відповідність за різного опису. Ці загальні механізми потрібно відрізняти від підтвердженої ручною перевіркою причини кожної конкретної помилки. Окремі технічні характеристики в цьому експерименті не вилучаються.

Після перегляду цих development-помилок зміни методу потребуватимуть нової незалежної перевірки. Наявні результати не є дозволом оптимізувати TEST або оголосити наукову новизну.
