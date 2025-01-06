# Multi-Armed Bandit

**Multi-Armed Bandit** is a Python library that provides implementations of algorithms for solving multi-armed bandit problems, including **Thompson Sampling** for various distributions.

## Project Structure
```
multi-armed-bandit/
│
├── bandits/                          # Core module for algorithm implementations
│   ├── __init__.py                   # Package initialization 
│   └── thompson_sampling.py          # Thompson Sampling implementation
│
├── experiments/                      # Examples and experiments
│   ├── __init__.py
│   ├── thompson_experiment.py        # Experiments with Thompson Sampling
│   └── other_experiments.py          # Experiments for other algorithms
│
├── tests/                            # Tests
│   ├── __init__.py
│   └── test_thompson_sampling.py     # Tests for Thompson Sampling
│
├── data/                             # Data for experiments
│   ├── sample_data.csv               # Example dataset
│   └── README.md                     # Dataset description
│
├── notebooks/                        # Jupyter notebooks for demonstration
│   └── demo_thompson_sampling.ipynb  # Demonstration of Thompson Sampling
│
├── setup.py                          # Script for package installation
├── requirements.txt                  # Project dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git exclusions
```