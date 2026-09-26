# Hyperparameter Optimisation with Genetic Algorithms

An investigation into optimising the hyperparameters of an Ensemble Gradient Boosting Regressor using a custom-built **Genetic Algorithm (GA)**, benchmarked against **Random Search (RS)** and **Bayesian Search (BS)**.

This project was originally submitted as an IB Computer Science Extended Essay (May 2025 session).

> **Research Question:** To what extent can genetic algorithms optimise the hyperparameters of an Ensemble Gradient Boosting Regressor model to predict housing prices compared to Bayes Search and Random Search?

---

## 1. Overview

Hyperparameter optimisation (HPO) is one of the trickiest parts of building a machine learning model. The wrong configuration can lead to under- or overfitting, wasted compute, or poor generalisation. This project implements a Genetic Algorithm **from scratch** and compares it against `RandomizedSearchCV`-style Random Search and Bayesian Search on a Gradient Boosting Regressor, using two real-world housing price datasets.

The GA draws on classic genetic algorithm theory (Holland, Goldberg, Mitchell, Grefenstette) and introduces one original mechanism, **dynamic population scaling**, that isn't standard in textbook GA implementations.

## 2. Datasets

| Dataset | Rows | Columns | Skew | Outliers | Source |
|---|---|---|---|---|---|
| California Housing | 20,640 | 7 | 0.978 (balanced) | 1,071 | Scikit-learn built-in dataset |
| Melbourne Housing | 8,895 | 360 | 2.412 (strong positive skew) | 420 | Kaggle (anthonypino/melbourne-housing-market) |

The two datasets were deliberately chosen for their contrasting characteristics. Melbourne is harder to model (low row-to-column ratio, heavy skew), while California is comparatively clean and balanced. A 70:30 train/test split was used throughout.

## 3. Model

**Gradient Boosting Regressor** (an ensemble of decision trees trained sequentially, each correcting the errors of the last) was chosen for its wide real-world use in price prediction and its large hyperparameter space, giving GA, BS, and RS plenty of room to differentiate.

### Hyperparameters optimised

| Hyperparameter | Small search space | Large search space |
|---|---|---|
| `n_estimators` | 150 – 300 | 100 – 600 |
| `learning_rate` | 0.1 – 0.3 | 0.01 – 0.3 |
| `max_depth` | 5 – 30 | 5 – 50 |
| `min_samples_split` | 4 – 10 | 2 – 20 |
| `min_samples_leaf` | 4 – 10 | 2 – 20 |
| `max_features` | 0.1 – 0.9 | 0.05 – 1.0 |
| `loss` | huber, squared_error, quantile | huber, squared_error, quantile |

The small search space contains ~1.55 × 10⁷ unique configurations; the large space ~7.18 × 10¹⁰ - roughly 5,000× larger, making exhaustive search infeasible for either.

## 4. How the Genetic Algorithm works

The GA encodes every hyperparameter configuration as a single binary **chromosome**, then evolves a population of these chromosomes over successive generations using selection, crossover, and mutation.

### Encoding
Each hyperparameter is encoded into a fixed-length block of bits and concatenated into one chromosome:
- **Integers** - signed binary (fixed-point coding).
- **Floats** - scaled-integer coding (multiplied by a power of 10 to preserve precision, then encoded as an integer).
- **Categorical strings** - each option is index-encoded and stored as an unsigned binary block.
- **Booleans** - a single bit.

This follows Goldberg's principles of *minimal alphabets* and *meaningful building blocks*, ensuring the chromosome stays compact without losing information on decoding.

### Selection - Rank-based selection + Roulette Wheel
Rather than raw fitness-proportionate selection (which can be skewed by outliers/scale), individuals are ranked by fitness and assigned a selection probability via:

```
Probability_rank(i) = [α + (rank(i)/(μ−1)) × (β − α)] / μ
```

where `α` and `β` control selection pressure (offspring allocated to the worst/best individual respectively). Parents are then chosen with roulette wheel selection weighted by these probabilities. Higher `β` → higher selection pressure → faster convergence; lower `β` → more exploration.

### Crossover - Hybrid single/multi-point
The chromosome is split at the boundaries between hyperparameter blocks, and each block undergoes single-point crossover independently. This respects the idea that each hyperparameter is its own "building block", swapping whole hyperparameter values between parents rather than cutting arbitrarily through the string.

### Mutation - Bit-flip
Each bit in the offspring chromosome has a small, independent chance (`mutation rate`) of flipping, which helps prevent premature convergence and lets the search escape local optima.

### Elitism
A fraction of the fittest individuals are copied unchanged into the next generation, guaranteeing that solution quality never regresses between generations.

### Dynamic population scaling *(original contribution)*
Population size shrinks over generations by "killing off" the worst-performing individuals, at a rate controlled by a `decay_rate`. This reduces the number of costly model evaluations needed as the population converges, which is particularly useful when each individual (i.e. each hyperparameter configuration) is expensive to train and test.

### Recommended GA parameters (from De Jong-style tuning)

| Parameter | Recommended value |
|---|---|
| Crossover rate | 0.7 |
| Mutation rate | 0.01 |
| Alpha rank (α) | 0.4 – 0.6 |
| Beta rank (β) | 1.4 – 1.6 |
| Elitism rate | 0.02 – 0.05 |
| Decay rate | 0 – 0.01 |
| Population size | 50 – 300 |
| Generation count | 25 – 100 |

## 5. Key Results

Fitness was measured as Mean Absolute Error (lower = better).

| Dataset | GA | BS | RS |
|---|---|---|---|
| Melbourne | **148,531.8** | 158,608.8 | 156,534.2 |
| California | **0.4458** | 0.4483 | 0.4472 |

**GA outperformed both alternatives on average**, with the biggest gains on the harder, skewed Melbourne dataset (+6.35% over BS, +5.11% over RS) and much smaller gains on the cleaner California dataset (+0.56% over BS, +0.31% over RS).

Other findings:
- **Larger search spaces favoured the GA even more**. The curse of dimensionality hurt Bayesian Search's surrogate model more than it hurt the GA's population-based search.
- **Bayesian Search was consistently the most time-efficient**, thanks to needing far fewer model evaluations, but it plateaued early and could be misled by skewed/outlier-heavy data.
- **Random Search was the least consistent**, occasionally beating GA/BS by chance but generally trailing both in accuracy and taking a comparable amount of time to GA despite using no logic.
- The GA's own final solution consistently improved 1.5%–5% over its best solution in the initial (random) generation, underscoring how much the genetic operators contribute beyond a lucky first guess.

**Bottom line:** if speed matters most, use Bayesian Search. If accuracy and robustness to messy/skewed data matter most, the GA is the stronger choice at the cost of significantly more compute time.

## 6. Repository / Appendix structure

```
├── README.md
├── GA_development
├──── GenHyperOptimizer.py - The core logic of the Genetic Algorithm 
├──── services.py - Helper functions: encoding/decoding, binary conversion, precision handling
├── GA_California / GA_Melbourne
├──── Iteration_(n) - Details of each generation for each run.
├──── f-(n)GA.py - Code for different configurations of the GA. 
├──── GA-(California/Melbourne)-report.txt - Summary statistics for each iteration run. 
├── Baseline_California / Baseline_Melbourne
├──── BayesSearch
├────── BayesSearchReport.txt - Summary statistics for each run. 
├────── f-(n)BayesSearch.py - Code for different configurations of the Bayes Search.
├────── f-(n)BayesSearchInfo.txt - Details of each Bayes Search run.
├──── RandomSearch 
├────── RandomSearchReport.txt - Summary statistics for each run. 
├────── f-(n)RandomSearch.py - Code for different configurations of the Random Search.
├────── f-(n)RandomSearchInfo.txt - Details of each Random Search run.
└── Box Plot.ipynb/Multiple Lines Plot.ipynb/Single Line Plot.ipynb - Notebooks to visualize results
```

### Basic usage sketch

```python
from GA_development.GenHyperOptimizer import GenHyperOptimizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

search_space = {
    "n_estimators": [150, 300],
    "learning_rate": [0.1, 0.3],
    "max_depth": [5, 30],
    "min_samples_split": [4, 10],
    "min_samples_leaf": [4, 10],
    "max_features": [0.1, 0.9],
    "loss": ["huber", "squared_error", "quantile"],
}

optimizer = GenHyperOptimizer(
    model=GradientBoostingRegressor,
    search_space=search_space,
    scoring=mean_absolute_error,
    objective="min",
    max_pop=60,
    max_gen=25,
    elitism_rate=0.05,
    reduction_rate=0.0,
)

best_params = optimizer.optimize(X_train, y_train, X_test, y_test, alpha_rank=0.4, beta_rank=1.6)
```

## 7. Limitations & Future Work

- The GA implementation is fairly basic - advanced operators such as **dominance** and **diploidy** were not explored.
- No **parallelisation** was implemented, despite the GA being parallel by nature (each individual could be evaluated independently).
- Currently, single-objective (minimising loss only); a **multi-objective** version incorporating training time, precision, and generalisation performance together would be more realistic for production use.
- The relationship between GA hyperparameters themselves (elitism rate, decay rate, crossover/mutation rate) and convergence behaviour deserves deeper study.
- Results are limited to one model (Gradient Boosting Regressor) and two datasets; the same methodology could extend to other models (neural networks, linear/tree models) and domains (finance, healthcare, climate).

## 8. References

Key sources used to design the GA (full citation list in the essay):
- Goldberg, D. E. - *Genetic Algorithms in Search, Optimization, and Machine Learning*
- Mitchell, M. - *An Introduction to Genetic Algorithms*
- Grefenstette, J. - *Rank-based Selection*, Handbook of Evolutionary Computation
- Frazier, P. I. - *A Tutorial on Bayesian Optimization*
- Yang, L. & Shami, A. - *On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice*

---

*Originally written as an IB Computer Science Extended Essay (May 2025 session).*