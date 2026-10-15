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
| softmax | 82.54% | 87.06% | 78.11% | 8.95% |
| detox | 50.00% | 50.00% | 50.00% | 0.00% |

![Fairness](outputs/figures/full/fairness_metrics.svg)

## Notes

- Paper also reports age-group + PPB; this lab twin focuses on gender under Protocol-1 with E-B=10%.
- See `DEVIATIONS.md`.
