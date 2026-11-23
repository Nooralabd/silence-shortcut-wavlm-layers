# Fixed conventions

These conventions hold for every encoder, condition and run.

## Labels, scores, layers

- Labels: `0` = bonafide, `1` = spoof. The score is the softmax probability of
  the spoof class; higher means more spoof-like. The sign is reversed before
  the official ASVspoof functions, which expect higher scores for bonafide.
- Layers are 1-based transformer layers. The output of the convolutional
  encoder (`hidden_states[0]`) is not used.

## Preprocessing

load → mono → resample to 16 kHz → (trimmed conditions only) trim → centre
crop to 64,000 samples if longer → peak normalisation `x / (max|x| + 1e-8)`.
The waveform is passed to the model directly; no feature-extractor
normalisation is applied.

## Silence trimming

`librosa.effects.trim` on the mono 16 kHz waveform, `frame_length=400`,
`hop_length=160`, threshold relative to the loudest frame of the file:
40 dB (`trim40`), 20 dB (`trim20`), 15 dB (`trim15`). Applied before the
crop. A file that is all zeros, or that would be shorter than 0.5 s (8,000
samples) after trimming, keeps its untrimmed audio and is flagged
`trim_fallback`.

librosa floors every frame RMS at 1e-5 before taking dB. In a file whose
loudest frame has an RMS below `1e-5 * 10 ** (top_db / 20)` (1e-3 at 40 dB),
no frame can be silent and nothing is trimmed. Trimming is applied before
peak normalisation, so this depends on the level of the recording.

## Extraction

- `unpadded` (primary): each file is encoded alone, at its own length inside
  the 4 s window, without zero padding and without an attention mask. A file
  with fewer than 400 samples is zero-padded to 400 and flagged `too_short`.
- `padded` (comparison): the file is zero-padded at the end to 64,000
  samples; pooling over all frames (`all-frames`) or over the frames of the
  real samples (`real-frames`).
- Pooling: mean over time and the population standard deviation (division
  by the number of frames). `stats` is their concatenation.
- The encoder is frozen, in evaluation mode, in float32.

## Classifier and training

- `Linear(D, 128) → ReLU → Dropout(0.3) → Linear(128, 2)`.
- Adam (learning rate 1e-4, weight decay 1e-5), batch size 128, 50 epochs,
  cosine annealing over the 50 epochs, cross-entropy weighted by
  `N / (2 N_c)`.
- The development-set EER of the training condition is computed after epoch 1
  and every 5 epochs; the checkpoint with the lowest value is kept (the
  earlier one on ties). This selection uses a ROC-based EER on the corner
  points of the ROC curve (`metrics.eer_roc_corners`); it is never reported.
- Seeds 42–46. One training function (`training.train_and_score`).
- One model per (encoder, extraction, training condition, layer, pooling,
  seed); it scores the evaluation set and In-the-Wild in every test
  condition of the design.

## Metrics

- EER: the official ASVspoof 2019 `compute_eer`.
- Tandem t-DCF: the official ASVspoof 2019 formulation with the official ASV
  scores and default cost parameters.
- All metrics are computed from saved scores. EER is computed per seed and
  then averaged; the spread is the sample standard deviation over seeds.

## Selection

Layers are selected by the mean development-set EER over seeds, per condition
and pooling (the lower layer on ties). Nothing is selected on the evaluation
set or on In-the-Wild.

## Statistics

- Bootstrap with 1,000 resamples, stratified by class, at the file level and
  at the speaker level (speakers drawn within class with all their files).
- In every resample the EER is computed per seed and averaged over seeds.
- 95% percentile intervals (2.5th and 97.5th percentiles). Differences are
  paired: both sides use the same resamples.
- Rank agreement between two 12-layer EER curves: Kendall's tau and
  Spearman's rho with speaker-level percentile and BCa intervals; exact
  two-sided Kendall p for 12 layers when neither curve has ties.
