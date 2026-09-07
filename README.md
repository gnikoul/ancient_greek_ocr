# Run
* inspect_eparchos.py
* prepare_eparchos.py
* count_eparchos.py
* analyze_eparchos.py
* split_eparchos.py
* check_unicode.py
* check_unicode_problem.py

kraken list
kraken get 10.5281/zenodo.22232579

python evaluate_baseline.py

### BASELINE RESULTS
Characters: 11175
Character errors: 2196
CER: 0.1965 (19.65%)

Words: 1901
Word errors: 1148
WER: 0.6039 (60.39%)


python train_lora_nrtr.py
python train_lora_pure.py
python train_lora_pure_v2.py

python evaluate_lora.py
...
Model                             CER         WER

Baseline                       19.65%      60.39%
lora_nrtr                      29.97%     107.36%
lora_only                      32.83%     109.15%
lora_only_cosine               31.37%     107.63%


python prepare_nfd_dataset.py
python verify_nfd_dataset.py

python train_lora_nfd.py

Baseline NFD                   16.81%      60.39%
Fine tuned                     5.35%      24.99%

python train_eparchos.py
python evaluate_baseline_nfd.py
python evaluate_xf53.py

python normalize_xf53_test.py

# fine tune on xf53
python extract_xf53_train_val.py
python normalize_xf53_train_val.py
python train_xf53.py
python compare_models_nfd.py

FINAL SUMMARY
Model          Dataset                 CER         WER
Base           EPARCHOS             16.81%      60.39%
Base           XF53                 24.31%      81.35%
EPARCHOS       EPARCHOS              5.35%      24.99%
EPARCHOS       XF53                 24.29%      86.66%
XF53           EPARCHOS             26.19%      82.69%
XF53           XF53                  9.60%      45.73%

# fine tune on both xf53 and eparchos
python combine_datasets.py
cat datasets/eparchos/training_nfd/val.lst datasets/XF53/val.lst > datasets/combined_val.lst
python train_combined.py
python compare_models_nfd.py

FINAL SUMMARY
Model          Dataset                 CER         WER
Base           EPARCHOS             16.81%      60.39%
Base           XF53                 24.31%      81.35%
EPARCHOS       EPARCHOS              5.35%      24.99%
EPARCHOS       XF53                 24.29%      86.66%
XF53           EPARCHOS             26.19%      82.69%
XF53           XF53                  9.60%      45.73%
EPARCHOS+XF53  EPARCHOS              5.50%      25.99%
EPARCHOS+XF53  XF53                  8.10%      40.14%


python detect_lines.py

python create_segmentation_dataset.py
python create_segmentation_splits
python train_segmentation.py
python compare_segmentation.py

python ocr_example.py