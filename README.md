# Overview

This is a playground in which i'm practicing with various ML algorithms and techniques. The main
goal is to further cement my understanding of the techniques i already know and to learn new ones.
I expect it will mostly be training simple QMNIST classifiers, the fun part will be in the
implementation of weird and wonderful models. Expect a lot of idiotic, failed experiments,
half-baked attempts at implementing poorly understood techniques from new, niche papers, and all
sorts of other nonsense. I will be trying to keep it organized, and keep a notebook in this readme
file. That being said: i absolutely do not recommend using this for examples of good code, it will
be more like the ramblings of a madman converted into code.

# Experiments

1. QMNIST classifier using a simple FFNN
2. QMNIST classifier using a CNN
3. QMNIST classifier using a transformer encoder
4. QMNIST classifier using a simple boosted tree

# Results

| Experiment | R² | Accuracy | RMS Error |
| - | - | - | - |
| 1 | 0.828241 | 0.913583 | 1.196773 |
| 2 | -0.291347 | 0.099283 | 3.281511 |
| 3 | -1.420184 | 0.113183 | 4.492381 |
| 4 | 0.946000 | 0.976233 | 0.671044 |

