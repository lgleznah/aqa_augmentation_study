# Functions for transforming the AVA ratings to various output formats
# This code has been adapted from code from ferrubio, in the private repository: https://github.com/ferrubio/AQA-framework

import numpy as np

# This function transforms the rating distribution to a 10-component probability distribution
def distribution_transform(scores):
    return (scores.T / np.sum(scores, axis=1).T).T

# This function transforms the rating distribution to a (0,1)-normalized average rating
def mean_transform(scores):
    return np.sum(scores * np.arange(0.1,1.1,0.1), axis=1) / np.sum(scores, axis=1)

# This function transforms the rating distribution to a binary label depending on whether the
# average rating is above-or-below 5
def binary_transform(scores):
    mean_scores = mean_transform(scores)
    classes = np.array(mean_scores >= 0.5).astype(int)
    return classes

# This function transforms the rating distribution to a Bernoulli distribution,
# using the result of mean_transform to generate [1-mean_transform, mean_transform]
def weights_transform(scores):
    mean_scores = mean_transform(scores)
    return np.vstack(([1-mean_scores],[mean_scores])).T

# Dummy function for Photozilla labels transformer function. The labels are already
# OK as they are in the CSV
def tenclass_transform(label):
    return label

# Dummy function for Photozilla OVR transformer function. Just fetch the label
def ovr_binary(label):
    return label