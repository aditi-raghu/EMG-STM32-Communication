# EMG-STM32-Communication
Real-time EMG-based Yes/No communication system using deep learning and STM32 TinyML deployment.

## sEMG preprocessing pipeline

Single-channel surface EMG, STM32 ADC, sensor RAW output,
about 325 Hz effective sample rate.

1. Keep Channel 4 (EMG). Channels 1, 2, 3, 5, 6 are flat.
2. Remove the first partial second, buffered bursts at the start, and the last partial second.
3. Make a time base from the sample index (fs estimated per file).
4. Remove the offset. Band-pass 20-150 Hz (Butterworth, order 4). Notch 50 and 100 Hz.
5. Moving RMS envelope (150 ms). Threshold-based event detection.
6. 250 ms windows, 125 ms step. Remove transition windows.
7. Features: MAV, RMS, WL, ZC, SSC, MNF, MDF, LF (raw power share < 20 Hz).
8. Normalize amplitude features to the rest level of each subject.
9. Leave-one-subject-out evaluation (LDA, random forest).

### Use

    conda create -n emg python=3.11 -y
    conda activate emg
    pip install -r requirements.txt
    python scripts/build_dataset.py
    python scripts/train_loso.py

### Known limits

- One channel on the inner forearm cannot separate "open" from "close".
  Add a second channel over the extensor muscles.
- The real sample rate is about 325 Hz, not 1000 Hz. Content above about 160 Hz is lost.
- No marker column. Labels come from the envelope threshold.
