# DEVIATIONS — Detox Loss

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Backbone | LightCNN-29 / ResNet-50 | ResNet-18 | Local compute |
| D2 | Tasks | Gender + age-group + PPB | Gender Protocol-1 (E-B 10%) | Focused lab twin |
| D3 | Clustering | Full-set / offline clusters | Mini-batch k-means | Online training |
| D4 | F-Distill layers | Multi-layer feature maps | Penultimate emb MSE vs ImageNet-init teacher | Simplified |
