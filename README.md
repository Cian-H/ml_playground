# Overview

This is a playground in which i'm practicing with various ML algorithms and techniques. The main
goal is to further cement my understanding of the techniques i already know and to learn new ones.
I expect it will mostly be training simple QMNIST classifiers. I will be trying to keep it
organized, and keep a notebook in this readme file. That being said: i absolutely do not recommend
using this for examples of good code, it will be more like the scribblings of a madman, converted
to half funcitoning code.

> [!Warning]
> This repo is intended to be my scratchpad for quick, messy testing of model architectures. They will not be implemented well. I will be moving fast and leaving a trail of destruction in here.

# Experiments

1. QMNIST classifier using a simple FFNN
2. QMNIST classifier using a CNN
3. QMNIST classifier using a transformer encoder
4. QMNIST classifier using a boosted decision tree classifier
5. QMNIST classifier using a RNN
6. QMNIST classifier using a boosted random forest classifier
7. QMNIST support vector classifier

# Results

| Experiment   |       R² |   Accuracy |   RMS Error |
|:-------------|---------:|-----------:|------------:|
| expt1        | 0.813407 |   0.911983 |    1.24738  |
| expt2        | 0.931031 |   0.972650 |    0.75826  |
| expt3        | 0.848160 |   0.934517 |    1.12524  |
| expt4        | 0.946000 |   0.976233 |    0.67104  |
| expt5        | 0.853349 |   0.933650 |    1.10585  |
| expt6        | 0.792609 |   0.894017 |    1.31507  |
| expt7        | 0.863442 |   0.934100 |    1.06711  |
