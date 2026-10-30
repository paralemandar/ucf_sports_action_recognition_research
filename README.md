# SlowFast Pathway Ablation on UCF Sports Action

Code and results for the paper:

> **Action recognition on the UCF Sports Action dataset using SlowFast networks: a statistically controlled ablation of pathway design**
> Mandar Parale, Shankar Mali, Chandrashekhar Patil
> School of Computer Engineering and Technology, MIT World Peace University, Pune, India

- **Kaggle notebook (runnable):** https://www.kaggle.com/code/mandarpparale/ucfsports-slowfast-exp
- **Dataset:** UCF Sports Action, https://www.crcv.ucf.edu/data/ucf_sports_actions.zip

---

## Overview

This repository reproduces every table and figure in the paper. It contains:

1. **SlowFast-Lite ablation.** A compact SlowFast network (1.11 M parameters) trained from scratch in seven configurations that vary the slow/fast channel ratio, the temporal window and the presence of each pathway.
2. **Pretrained baselines.** Seven Kinetics-400 backbones (R(2+1)D-18, S3D, SlowFast-R50, MViTv2-S, Video Swin-T, TimeSformer-B, VideoMAE-B), evaluated as frozen feature extractors with a logistic-regression classifier.
3. **Statistical evaluation.** Stratified 5-fold cross-validation on all 150 videos, using the same folds for every model. Models are compared with exact McNemar tests (Holm-corrected) and paired bootstrap 95% confidence intervals.

## Main results (5-fold CV, 150 videos, 13 classes)

| Model | Accuracy (%) | Macro-F1 (%) | Holm-corrected p vs. Baseline |
|---|---|---|---|
| SlowFast-Lite Baseline | 62.67 ± 8.30 | 54.77 ± 8.84 | – |
| Fast-Dominant | 61.33 ± 7.67 | 51.72 ± 7.43 | 1.00 |
| Slow-Dominant | 63.33 ± 7.07 | 53.80 ± 6.50 | 1.00 |
| Small-Window | 64.67 ± 4.47 | 57.20 ± 5.36 | 1.00 |
| Large-Window | 60.67 ± 7.96 | 53.07 ± 7.38 | 1.00 |
| Slow-Only | 63.33 ± 6.67 | 55.50 ± 7.19 | 1.00 |
| Fast-Only | 40.00 ± 4.71 | 21.56 ± 2.64 | < 0.001 |
| S3D (Kinetics-400, linear probe) | **88.67 ± 5.58** | **86.69 ± 7.15** | < 0.001 |

Full results for all 14 models are in `results/tables/`.

## Repository structure

```
├── ucf_sports_slowfast_kaggle.ipynb   # main notebook (run on Kaggle)
├── ucf_sports_slowfast_kaggle.py      # same code as a plain Python script
├── results/
│   ├── config.json                    # all hyperparameters and configuration definitions
│   ├── environment.json               # library versions and GPU used
│   ├── folds.csv                      # fold assignment of every video (seed 42)
│   ├── video_index.csv                # video list, class and frame count
│   ├── predictions/                   # per-video predictions, one CSV per model × fold
│   ├── histories/                     # per-epoch training logs
│   ├── tables/                        # main results, significance tests, per-class metrics (CSV + LaTeX)
│   └── figures/                       # figures (PDF + PNG)
└── README.md
```

The UCF Sports videos are **not** included. Please download them from the official source above.

## How to reproduce (Kaggle, recommended)

1. **Prepare the dataset.** Download `ucf_sports_actions.zip` from the link above. On Kaggle, go to **Datasets → New Dataset** and upload the zip. Kaggle extracts it automatically. The expected layout is:
   ```
   ucf_sports_actions/ucf action/<Class>/<video_id>/*.jpg
   ```
   The notebook finds the 13 class folders anywhere under `/kaggle/input`.
2. **Open the notebook:** https://www.kaggle.com/code/mandarpparale/ucfsports-slowfast-exp, then click **Copy & Edit**.
3. **Attach the dataset:** **Add Input** → select the dataset you uploaded in step 1.
4. **Settings:** set **Accelerator** to **GPU T4** and turn **Internet** **On**. Internet is needed only to download the pretrained weights.
5. **Run all:** **Run → Run All**. Results are written to `/kaggle/working/UCF_Sports_Action/results/`.
6. **Download:** after the run, click **Save Version → Save & Run All**, then use **Output → Download All**.

**Runtime.** On one NVIDIA T4, each SlowFast-Lite run takes about 5 minutes. The 7 configurations × 5 folds take about 3 hours, and the pretrained baselines about 30 minutes.

**Resuming.** Every finished run is saved immediately. If a session stops, attach the previous output as an input and set `resume_from` in the configuration cell; completed runs are skipped.

### Running locally

```bash
pip install torch torchvision opencv-python scikit-learn scipy pandas matplotlib transformers fvcore
# set CFG["data_root"] to the extracted dataset folder and CFG["work_dir"] to an output folder
python ucf_sports_slowfast_kaggle.py
```

## Experimental protocol

| Setting | Value |
|---|---|
| Videos / classes | 150 / 13 (view-specific labels from the archive folders) |
| Evaluation | Stratified 5-fold CV, fold seed 42, identical folds for all models |
| Input clip | 32 frames, 112 × 112, sampled uniformly from a window of W frames |
| Training | AdamW, lr 1e-3, weight decay 1e-4, batch 8, 50 epochs, lr × 0.5 at epochs 20 and 40 |
| Augmentation | Scale jitter, random crop, horizontal flip, brightness/contrast jitter, random window position |
| Checkpoint | Final epoch (no model selection on test data) |
| Test-time | One centre clip, centre crop |
| Statistics | Exact McNemar test + Holm correction; paired bootstrap 95% CI (10,000 resamples) |
| Seed | 0 (set `seeds=[0, 1, 2]` in the configuration cell to add more) |

## Configurations

| Configuration | α | T fast | T slow | Window | C slow | C fast | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|---|---|
| Baseline | 4 | 32 | 8 | 64 | 22 | 3 | 1.11 | 1.26 |
| Fast-Dominant | 4 | 32 | 8 | 64 | 11 | 11 | 0.79 | 2.72 |
| Slow-Dominant | 4 | 32 | 8 | 64 | 24 | 2 | 1.23 | 1.18 |
| Small-Window | 4 | 32 | 8 | 32 | 22 | 3 | 1.11 | 1.26 |
| Large-Window | 4 | 32 | 8 | whole video | 22 | 3 | 1.11 | 1.26 |
| Slow-Only | 4 | – | 8 | 64 | 22 | – | 0.93 | 0.68 |
| Fast-Only | – | 32 | – | 64 | – | 3 | 0.02 | 0.49 |

## Citation

If you use this code, please cite:

```bibtex
@article{parale2026slowfast,
  title   = {Action recognition on the UCF Sports Action dataset using SlowFast networks: a statistically controlled ablation of pathway design},
  author  = {Parale, Mandar and Mali, Shankar and Patil, Chandrashekhar},
  journal = {Pattern Recognition Letters},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the dataset:

```bibtex
@inproceedings{rodriguez2008action,
  title     = {Action {MACH}: A spatio-temporal maximum average correlation height filter for action recognition},
  author    = {Rodriguez, Mikel D. and Ahmed, Javed and Shah, Mubarak},
  booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1--8},
  year      = {2008}
}
@incollection{soomro2014action,
  title     = {Action recognition in realistic sports videos},
  author    = {Soomro, Khurram and Zamir, Amir R.},
  booktitle = {Computer Vision in Sports},
  publisher = {Springer},
  pages     = {181--208},
  year      = {2014}
}
```

## License

Code: [MIT License]. The UCF Sports Action dataset is provided by the Center for Research in Computer Vision, University of Central Florida, under its own terms of use and is not redistributed here.

## Contact

Shankar Mali (corresponding author): shankar.mali@mitwpu.edu.in
School of Computer Engineering and Technology, MIT World Peace University, Pune, India
