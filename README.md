# HyPersona

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

A semi-automated framework for persona development and the hyperparameter tuning of clustering algorithms.
This code accompanies the publication: [Selecting a clustering algorithm: A semi-automated hyperparameter tuning framework for effective persona development](https://doi.org/10.1016/j.array.2022.100186)

## Motivation

Selecting the algorithm and parameters to use for a clustering problem, known as hyperparameter tuning, can be difficult.
HyPersona aims to similify the hyperparameter tuning process by applying an exhaustive grid search across a series of clustering algorithms and parameters, and uses thresholds to automatically rule out cluster sets. 
HyPersona then develops a series of graphs and CSV files to facilitate the selection of a clustering algorithm and parameters.

## Installation 

HyPersona is currently not available for installation as a library, however the source code can be downloaded and used. 

_TODO: set up with pip_

HyPersona requires:
- [sklearn](https://scikit-learn.org/stable/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)


## Example Usage 

_TODO: add examples_

## Contributing Guidelines

Please make a [pull request](https://github.com/ElizabethForest/HyPersona/pulls) or an [issue](https://github.com/ElizabethForest/HyPersona/issues) if you would like to contribute or have any bug reports, issues, or suggestions. 

If you contribute... 

_TODO: add testing and contribution guidelines_
