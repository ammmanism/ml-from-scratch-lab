
import json
import os

nb_path = r"c:\Users\amman\.gemini\antigravity\scratch\ml_from_scratch_lib\notebooks\00_math_foundations\probability_distributions.ipynb"

# --- Helper Functions ---
def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

# --- Content Definitions ---

cells = []

# 1. Header & Title
cells.append(create_markdown_cell("""# 🎲 Probability Distributions
**Mathematical Foundations for Machine Learning**

> **Author**: Elite ML Educator
> **Goal**: To provide a deep, intuitive, and mathematical understanding of probability distributions used in ML.

## 📹 Recommended Video Lectures
> 💡 **Pro Tip**: Watch these videos to build intuition before diving into the math and code."""))

cells.append(create_code_cell("""from IPython.display import YouTubeVideo

# StatQuest: Probability Distributions
YouTubeVideo('oI3hZJqXJuc', width=800, height=450)"""))

cells.append(create_markdown_cell("""---
## 1. Introduction 📚

### 🧠 What is a Probability Distribution?
A **probability distribution** is a mathematical function that describes the likelihood of obtaining the possible values that a random variable can assume.

In Machine Learning, understanding distributions is critical for:
*   **Data Analysis**: Understanding the underlying structure of your data.
*   **Model Building**: Assumptions like "errors are normally distributed" (Linear Regression).
*   **Generative Models**: GANs and VAEs learn to map distributions.

### 🔑 Key Concepts

| Term | Symbol | Definition |
| :--- | :---: | :--- |
| **Random Variable** | $X$ | A variable whose value is subject to variations due to chance. |
| **Probability Mass Function** | **PMF** $P(X=x)$ | Probability that a *discrete* random variable is exactly equal to some value. |
| **Probability Density Function** | **PDF** $f(x)$ | Relative likelihood that a *continuous* random variable takes a value near $x$. |
| **Cumulative Distribution Function** | **CDF** $F(x)$ | Probability that $X$ will take a value less than or equal to $x$. |

> ⚠️ **Note**: For continuous distributions, $P(X=x) = 0$. We calculate probabilities over intervals: $P(a \le X \le b) = \int_a^b f(x) dx$.
"""))

# 2. Setup
cells.append(create_markdown_cell("### ⚙️ Setup & Imports"))
cells.append(create_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Elite Visual Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

%matplotlib inline"""))

# 3. Discrete Distributions
cells.append(create_markdown_cell("""---
## 2. Discrete Distributions 🧱

Discrete distributions model variables that have a countable number of outcomes (integers).
Examples: Coin flips, rolling dice, number of emails received.

### 2.1 🪙 Bernoulli Distribution
The simplest discrete distribution. Models a **single trial** with two possible outcomes:
*   **Success** ($k=1$) with probability $p$
*   **Failure** ($k=0$) with probability $1-p$ (often denoted $q$)

> 🔢 **Formula**:
> $$ P(X=k) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0, 1\} $$

**Properties**:
*   **Mean**: $E[X] = p$
*   **Variance**: $Var(X) = p(1-p)$
"""))

cells.append(create_code_cell("""# Bernoulli Parameters
p = 0.6  # Probability of success (e.g., loaded coin)

# Generate variates
bernoulli_dist = stats.bernoulli(p)
x = [0, 1]
pmf = [bernoulli_dist.pmf(k) for k in x]

# Visualization
plt.figure(figsize=(10, 5))
bars = plt.bar(x, pmf, color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', width=0.6)
plt.xticks(x, ['Failure (0)', 'Success (1)'])
plt.ylabel('Probability')
plt.title(f'Bernoulli Distribution PMF (p={p})')
plt.ylim(0, 1)

# Annotate
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom')
plt.show()"""))

cells.append(create_markdown_cell("""### 2.2 🔢 Binomial Distribution
Generalization of Bernoulli. Models the number of successes $k$ in **$n$ independent** Bernoulli trials.

> 🔢 **Formula**:
> $$ P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} $$

**Parameters**:
*   $n$: Number of trials.
*   $p$: Probability of success in each trial.

> 🧩 **Real World Example**: If you flip a fair coin 10 times, what is the probability of getting exactly 7 heads?
"""))

cells.append(create_code_cell("""# Binomial Parameters
n_trials = 20
p_success = 0.5  # Fair coin

# X-axis: Number of successes (0 to n)
k_values = np.arange(0, n_trials + 1)
binom_pmf = stats.binom.pmf(k_values, n_trials, p_success)

# Visualization
plt.figure(figsize=(12, 6))
plt.vlines(k_values, 0, binom_pmf, colors='navy', lw=5, alpha=0.5)
plt.plot(k_values, binom_pmf, 'bo', ms=8, label=f'n={n_trials}, p={p_success}')
plt.title('Binomial Distribution PMF')
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability P(X=k)')
plt.xticks(np.arange(0, n_trials+1, 2))
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.show()"""))

cells.append(create_markdown_cell("""#### 🧪 Experiment: Effect of 'n' and 'p'
Let's see how changing the probability of success shifts the distribution.
"""))

cells.append(create_code_cell("""plt.figure(figsize=(14, 7))
n = 20
for p_val, color in zip([0.1, 0.5, 0.9], ['red', 'green', 'blue']):
    pmf = stats.binom.pmf(k_values, n, p_val)
    plt.plot(k_values, pmf, '-o', label=f'p={p_val}', color=color, alpha=0.7)

plt.title(f'Effect of "p" on Binomial Distribution (n={n})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.legend()
plt.show()"""))

cells.append(create_markdown_cell("""### 2.3 📞 Poisson Distribution
Models the number of events occurring in a fixed interval of time or space.
**Assumptions**: Events occur with a known constant mean rate and independently of the time since the last event.

> 🔢 **Formula**:
> $$ P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

**Parameter**:
*   $\lambda$ (Lambda): Average number of events per interval. (Also equal to Variance!).

> 🧠 **Key Concept**: As $n \to \infty$ and $p \to 0$ in a Binomial distribution, it converges to Poisson (Law of Rare Events).
"""))

cells.append(create_code_cell("""# Poisson Parameters
lambdas = [1, 4, 10]
k_axis = np.arange(0, 20)

plt.figure(figsize=(14, 7))

for l in lambdas:
    pmf = stats.poisson.pmf(k_axis, l)
    plt.plot(k_axis, pmf, '-o', ms=6, label=f'$\lambda={l}$')

plt.title('Poisson Distribution PMF')
plt.xlabel('Number of Events (k)')
plt.ylabel('Probability P(X=k)')
plt.legend()
plt.xticks(k_axis)
plt.show()"""))

# 4. Continuous Distributions
cells.append(create_markdown_cell("""---
## 3. Continuous Distributions 🌊

Continuous distributions model variables that can take any value within a range (infinite possibilities).
Examples: Height, Time, Temperature, Stock Prices.

### 3.1 ⏹️ Uniform Distribution
All outcomes in the range $[a, b]$ are equally likely. "Maximum Entropy" distribution when only bounds are known.

> 🔢 **Formula (PDF)**:
> $$ f(x) = \frac{1}{b-a} \quad \text{for } a \le x \le b $$
"""))

cells.append(create_code_cell("""# Uniform Parameters
a, b = 0, 10
x = np.linspace(a-2, b+2, 1000)
pdf = stats.uniform.pdf(x, loc=a, scale=b-a)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf, lw=3, color='purple')
plt.fill_between(x, pdf, color='purple', alpha=0.2)
plt.title(f'Continuous Uniform Distribution U({a}, {b})')
plt.ylabel('Density')
plt.xlabel('Values')
plt.ylim(0, 0.2)
plt.show()"""))

cells.append(create_markdown_cell("""### 3.2 🔔 Normal (Gaussian) Distribution
The **King** of distributions. By the **Central Limit Theorem**, sums of independent random variables tend toward a Normal distribution. It appears everywhere with natural variations.

> 🔢 **Formula (PDF)**:
> $$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} $$

**Parameters**:
*   $\mu$ (Mu): Mean (Center).
*   $\sigma$ (Sigma): Standard Deviation (Spread).
"""))

cells.append(create_code_cell("""# Normal Parameters
mu_values = [0, 0, -2]
sigma_values = [1, 2, 0.5]
colors = ['blue', 'orange', 'green']

x = np.linspace(-6, 6, 1000)

plt.figure(figsize=(14, 7))

for mu, sigma, c in zip(mu_values, sigma_values, colors):
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    plt.plot(x, pdf, label=f'$\mu={mu}, \sigma={sigma}$', color=c, lw=2)
    plt.fill_between(x, pdf, color=c, alpha=0.1)

plt.title('Normal (Gaussian) Distribution PDF')
plt.ylabel('Density')
plt.xlabel('x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""))

cells.append(create_markdown_cell("""#### 📉 The 68-95-99.7 Rule
For a Normal distribution:
*   **68%** of data falls within $\mu \pm 1\sigma$
*   **95%** of data falls within $\mu \pm 2\sigma$
*   **99.7%** of data falls within $\mu \pm 3\sigma$
"""))

# 5. Advanced Concepts
cells.append(create_markdown_cell("""---
## 4. Advanced Concepts 🎓

### 4.1 🤝 Central Limit Theorem (CLT) Demo
> ****Theorem****: The distribution of the *sample mean* tends toward a Normal distribution as the sample size increases, regardless of the original population's distribution.

Let's prove this by sampling from a **Uniform** distribution (which looks nothing like a bell curve) and plotting the means.
"""))

cells.append(create_code_cell("""# Parameters
pop_size = 100000
sample_size = 50   # n
num_samples = 1000 # Number of times we calculate the mean

# 1. Create a non-normal population (Uniform)
population = np.random.uniform(0, 100, pop_size)

# 2. Draw samples and calculate means
sample_means = []
for _ in range(num_samples):
    sample = np.random.choice(population, sample_size)
    sample_means.append(np.mean(sample))

# Visualization
plt.figure(figsize=(16, 6))

# Plot Population
plt.subplot(1, 2, 1)
plt.hist(population, bins=50, color='gray', alpha=0.5, density=True)
plt.title('Original Population (Uniform)')
plt.ylabel('Density')

# Plot Sample Means
plt.subplot(1, 2, 2)
sns.histplot(sample_means, kde=True, color='crimson')
plt.title(f'Distribution of Sample Means (n={sample_size})')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')

plt.suptitle('Central Limit Theorem in Action 🚀', fontsize=20)
plt.tight_layout()
plt.show()"""))

# 6. Exercises
cells.append(create_markdown_cell("""---
## 5. Exercises ✍️

> 💡 **Tip**: Use `scipy.stats` for calculations.

### Exercise 1: Quality Control 🏭
A factory produces light bulbs where 2% are defective. You randomly sample 50 bulbs.
1. What distribution models this scenario?
2. What is the probability that exactly 3 are defective?
3. What is the probability that *at least* 1 is defective?
"""))

cells.append(create_code_cell("""# Write your code here
# Hint: n=50, p=0.02. Use Binomial.
"""))

cells.append(create_markdown_cell("""### Exercise 2: Website Traffic 🌐
A website gets an average of 10 hits per minute.
1. What is the probability of getting exactly 15 hits in a minute?
2. What is the probability of getting fewer than 5 hits?
"""))

cells.append(create_code_cell("""# Write your code here
# Hint: lambda=10. Use Poisson.
"""))

# 7. Summary
cells.append(create_markdown_cell("""---
## 6. Summary & Key Takeaways 📝

*   **Discrete vs Continuous**: Countable outcomes (PMF) vs Infinite range (PDF).
*   **Bernoulli/Binomial**: Success/Failure trials.
*   **Poisson**: Events over time/space (rare events).
*   **Normal**: The "Bell Curve" that appears everywhere due to the **CLT**.
*   **CLT**: Sample means become Normal, empowering most statistical inference methods.

## 📚 Further Reading
*   [Bishop, C. M. "Pattern Recognition and Machine Learning" (Chapter 2)](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
*   [Khan Academy: Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
"""))

# --- Write to File ---
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated Elite Probability Notebook at: {nb_path}")
