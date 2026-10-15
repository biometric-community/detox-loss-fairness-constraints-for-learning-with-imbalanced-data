# REPORT — Detox Loss: Fairness Constraints for Imbalanced Data

**Protocol:** `full`

Lab twin: gender classification on UTKFace Protocol-1 (E-B ratio=0.1); Softmax vs Detox (Exclusion+Inclusion+F-Distill).

## Training

- Task: gender
- Train images: 3108 ({'ea': 2797, 'eb': 311, 'total': 3108})
- Methods: softmax, detox

![Training](outputs/figures/full/train_loss.svg)

## Fairness evaluation (balanced test)

| Method | BAcc | E-A BAcc | E-B BAcc | |BAcc gap| |
|--------|------|----------|----------|-----------|
| softmax | 87.94% | 90.10% | 85.82% | 4.28% |
| detox | 72.82% | 78.65% | 67.23% | 11.42% |

![Fairness](outputs/figures/full/fairness_metrics.svg)

## Notes

- Paper also reports age-group + PPB; this lab twin focuses on gender under Protocol-1 with E-B=10%.
- See `DEVIATIONS.md`.
