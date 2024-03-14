# Project 2: Naive Bayes

## Introduction
The Naïve Bayes Project is a gender classification system designed to predict gender based on personal names. The system is built using the Naïve Bayes algorithm and can differentiate between typically male and female names.

## Key Components
- `naivebayesPXY.py`: This script computes the conditional probabilities P(X|Y), which are fundamental to the functioning of the Naïve Bayes classifier.
- `naivebayesPY.py`: Responsible for estimating the class probabilities P(Y), determining the likelihood of each class within the training set.
- `naivebayes.py`: Calculates the log probabilities log P(Y|X = x1) using Bayes' Rule, central to decision-making in the classifier.
- `naivebayesCL.py`: Transforms the output of the Naïve Bayes algorithm into a format suitable for use with a linear classifier.
- `classifyLinear.py`: Applies the linear classifier to the dataset to make final gender predictions.
