
import json
import os

nb_path = r"c:\Users\amman\.gemini\antigravity\scratch\ml_from_scratch_lib\notebooks\00_math_foundations\probability_distributions.ipynb"

# --- Content Definitions (Same as before, ensuring keys are robust) ---

hero_cell = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class='hero-header'>\n",
    "    <div class='hero-title'>Probability Distributions</div>\n",
    "    <div class='hero-subtitle'>Mathematical Foundations for Machine Learning 🤖 📊</div>\n",
    "</div>\n",
    "\n",
    "<style>\n",
    "    /* Google Fonts */\n",
    "    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Fira+Code:wght@500&display=swap');\n",
    "\n",
    "    :root {\n",
    "        --primary: #6366f1; /* Indigo 500 */\n",
    "        --secondary: #ec4899; /* Pink 500 */\n",
    "        --accent: #8b5cf6; /* Violet 500 */\n",
    "        --bg-color: #f8fafc; /* Slate 50 */\n",
    "        --text-color: #1e293b; /* Slate 800 */\n",
    "        --code-bg: #1e293b;\n",
    "        --code-text: #e2e8f0;\n",
    "    }\n",
    "\n",
    "    body {\n",
    "        font-family: 'Inter', sans-serif !important;\n",
    "        color: var(--text-color) !important;\n",
    "    }\n",
    "\n",
    "    /* Hero Header */\n",
    "    .hero-header {\n",
    "        background: linear-gradient(135deg, var(--primary), var(--secondary));\n",
    "        color: white !important;\n",
    "        padding: 2rem;\n",
    "        border-radius: 1rem;\n",
    "        margin-bottom: 2rem;\n",
    "        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);\n",
    "    }\n",
    "    .hero-title {\n",
    "        font-size: 2.5rem;\n",
    "        font-weight: 800;\n",
    "        margin-bottom: 0.5rem;\n",
    "        color: white !important;\n",
    "    }\n",
    "    .hero-subtitle {\n",
    "        font-size: 1.2rem;\n",
    "        opacity: 0.9;\n",
    "        color: white !important;\n",
    "    }\n",
    "\n",
    "    /* Callout Boxes */\n",
    "    .callout {\n",
    "        padding: 1.5rem;\n",
    "        border-radius: 0.75rem;\n",
    "        margin: 1.5rem 0;\n",
    "        border-left: 5px solid;\n",
    "        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);\n",
    "    }\n",
    "    .callout-def {\n",
    "        background-color: #eff6ff;\n",
    "        border-color: #3b82f6;\n",
    "    }\n",
    "    .callout-concept {\n",
    "        background-color: #f5f3ff;\n",
    "        border-color: #8b5cf6;\n",
    "    }\n",
    "    .callout-formula {\n",
    "        background-color: #1e293b;\n",
    "        color: #f8fafc !important;\n",
    "        border-color: var(--secondary);\n",
    "        font-family: 'Fira Code', monospace;\n",
    "        text-align: center;\n",
    "        font-size: 1.1rem;\n",
    "    }\n",
    "    .callout-warning {\n",
    "        background-color: #fff7ed;\n",
    "        border-color: #f97316;\n",
    "    }\n",
    "</style>\n",
    "\n",
    "<div class='callout callout-concept'>\n",
    "    <h3>🌟 Overview</h3>\n",
    "    <p>This comprehensive notebook explores probability distributions, their properties, and applications in machine learning. We will visually and conducting mathematically rigor analysis of:</p>\n",
    "    <ul>\n",
    "        <li><strong>Discrete Distributions</strong> (Bernoulli, Binomial, Poisson, etc.)</li>\n",
    "        <li><strong>Continuous Distributions</strong> (Normal, Exponential, Beta, etc.)</li>\n",
    "        <li><strong>Key Statistical Measures</strong> (Entropy, Moments)</li>\n",
    "        <li><strong>ML Applications</strong> (Naive Bayes, MLE, Conjugate Priors)</li>\n",
    "    </ul>\n",
    "</div>"
   ]
}

intro_discrete = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Introduction to Probability Distributions 🎲\n",
    "\n",
    "<div class='callout callout-def'>\n",
    "    <h3>📘 Definition</h3>\n",
    "    A <strong>probability distribution</strong> provides a mathematical framework for quantifying uncertainty, describing how probabilities are distributed over the values of a random variable.\n",
    "</div>\n",
    "\n",
    "### 🗝️ Key Concepts\n",
    "\n",
    "| Concept | Symbol | Description |\n",
    "|---------|:------:|-------------|\n",
    "| **Random Variable** | $X$ | A variable whose possible values are outcomes of a random phenomenon. |\n",
    "| **PMF** | $p(x)$ | **Probability Mass Function**: For discrete variables, $P(X = x)$. |\n",
    "| **PDF** | $f(x)$ | **Probability Density Function**: For continuous variables, relative likelihood. |\n",
    "| **CDF** | $F(x)$ | **Cumulative Distribution Function**: $P(X \\leq x)$. |\n",
    "\n",
    "<div class='callout callout-formula'>\n",
    "    <div>Discrete: $$ P(X = x) = p(x) $$</div>\n",
    "    <div>Continuous: $$ P(a \\leq X \\leq b) = \\int_a^b f(x) dx $$</div>\n",
    "    <div>Cumulative: $$ F(x) = P(X \\leq x) = \\int_{-\\infty}^x f(t) dt $$</div>\n",
    "</div>\n",
    "\n",
    "## 2. Discrete Probability Distributions\n",
    "Discrete distributions model outcomes that take countable integer values (e.g., number of heads, number of customers)."
   ]
}

intro_continuous = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Continuous Probability Distributions 🌊\n",
    "Continuous distributions model variables that can take any value within a range (e.g., height, time, temperature)."
   ]
}

# Broader Keys for Matching
dist_map = {
    "Bernoulli": [
        "### 2.1 🪙 Bernoulli Distribution\n",
        "The simplest discrete distribution, modeling a single trial with two possible outcomes (success/failure).\n",
        "<div class='callout callout-formula'>$$ P(X=k) = p^k(1-p)^{1-k}, \\quad k \\in \\{0, 1\\} $$</div>\n",
        "| Param | Symbol | Meaning |\n",
        "|---|---|---|\n",
        "| Probability | $p$ | Probability of success |\n"
    ],
    "Binomial": [
        "### 2.2 🔢 Binomial Distribution\n",
        "Models the number of successes in $n$ independent Bernoulli trials.\n",
        "<div class='callout callout-formula'>$$ P(X=k) = \\binom{n}{k} p^k (1-p)^{n-k} $$</div>\n",
        "| Param | Symbol | Meaning |\n",
        "|---|---|---|\n",
        "| Trials | $n$ | Number of independent trials |\n",
        "| Probability | $p$ | Probability of success in each trial |\n"
    ],
    "Poisson": [
        "### 2.3 📞 Poisson Distribution\n",
        "Models the number of events occurring in a fixed interval of time/space.\n",
        "<div class='callout callout-formula'>$$ P(X=k) = \\frac{\\lambda^k e^{-\\lambda}}{k!} $$</div>\n",
        "| Param | Symbol | Meaning |\n",
        "|---|---|---|\n",
        "| Rate | $\\lambda$ | Average number of events per interval |\n"
    ],
    "Geometric": [
        "### 2.4 🎯 Geometric Distribution\n",
        "Models the number of trials needed to get the *first* success.\n",
        "<div class='callout callout-formula'>$$ P(X=k) = (1-p)^{k-1}p $$</div>\n",
        "| Param | Symbol | Meaning |\n",
        "|---|---|---|\n",
        "| Probability | $p$ | Probability of success |\n"
    ],
    "Uniform": [
        "### 3.1 ⏹️ Uniform Distribution\n",
        "All variability is equally likely within the range $[a, b]$.\n",
        "<div class='callout callout-formula'>$$ f(x) = \\frac{1}{b-a}, \\quad a \\le x \\le b $$</div>\n"
    ],
    "Normal": [
        "### 3.2 🔔 Normal (Gaussian) Distribution\n",
        "The most important distribution in statistics (Central Limit Theorem).\n",
        "<div class='callout callout-formula'>$$ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2} $$</div>\n",
        "| Param | Symbol | Meaning |\n",
        "|---|---|---|\n",
        "| Mean | $\\mu$ | Center of the distribution |\n",
        "| Std Dev | $\\sigma$ | Spread/Width |\n"
    ],
    "Exponential": [
        "### 3.3 ⏱️ Exponential Distribution\n",
        "Models the time between events in a Poisson process.\n",
        "<div class='callout callout-formula'>$$ f(x) = \\lambda e^{-\\lambda x} $$</div>\n"
    ],
    "Gamma": [
        "### 3.4 visit Gamma Distribution\n",
        "Generalization of Exponential/Chi-squared. Models waiting times for $k$ events.\n",
        "<div class='callout callout-formula'>$$ f(x) = \\frac{x^{k-1}e^{-x/\\theta}}{\\Gamma(k)\\theta^k} $$</div>\n"
    ],
    "Beta": [
        "### 3.5 📊 Beta Distribution\n",
        "Defined on $[0, 1]$, often used as a prior for probabilities (conjugate to Binomial).\n",
        "<div class='callout callout-formula'>$$ f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha, \\beta)} $$</div>\n"
    ],
    "Chi-square": [
        "### 3.6 🧪 Chi-Squared Distribution\n",
        "Sum of squares of $k$ independent standard normal variables. Vital for hypothesis testing.\n"
    ],
    "Student's t": [
        "### 3.7 🎓 Student's t-Distribution\n",
        "Used when estimating mean of normally distributed population with unknown $\\sigma$ (small samples).\n"
    ],
    "F-distribution": [
        "### 3.8 📈 F-Distribution\n",
        "Ratio of two Chi-squared variates. Used in ANOVA.\n"
    ],
    "Log-normal": [
        "### 3.9 🪵 Log-Normal Distribution\n",
        "Variable whose logarithm is normally distributed.\n"
    ],
    "Laplace": [
        "### 3.10 🏔️ Laplace Distribution\n",
        "Double exponential distribution. Used in Lasso regression (L1 regularization).\n"
    ],
    "Comparing multiple": [
        "## 4. Distribution Relationships & Comparisons 🔄\n",
        "Visualizing how different distributions relate to each other."
    ],
    "Bayes": [
        "## 5. Bayesian Inference 🧠\n",
        "Using distributions as priors and posteriors.\n",
        "<div class='callout callout-concept'>$$ P(\\theta | X) \\propto P(X | \\theta) P(\\theta) $$</div>\n"
    ],
    "Central Limit": [
        "## 6. Central Limit Theorem (CLT) 📉\n",
        "Demonstrating that sums of independent variables tend toward a Normal distribution.\n"
    ],
    "Entropy": [
        "## 7. Information Theory Measures ℹ️\n",
        "Quantifying uncertainty and distance between distributions."
    ]
}

def create_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

# --- Execution ---

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Step 1: Filter out existing enhancements (Idempotency)
clean_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source_str = "".join(cell['source'])
        # Identify our custom styling tags
        if "hero-header" in source_str or "callout" in source_str or "Google Fonts" in source_str:
            continue # Skip our own generated cells
        
        # Establish robust filtering for ALL generated headers
        if "## 1. Introduction to Probability Distributions" in source_str:
            continue
        if "## 2. Discrete Probability Distributions" in source_str:
            continue
        if "## 3. Continuous Probability Distributions" in source_str:
            continue
        if "## 4. Distribution Relationships" in source_str:
            continue
        if "## 5. Bayesian Inference" in source_str:
            continue
        if "## 6. Central Limit Theorem" in source_str:
            continue
        if "## 7. Information Theory Measures" in source_str:
            continue

        # Also skip the old generic headers we intentionally replaced
        if "## Discrete Distributions" in source_str or "## Continuous Distributions" in source_str:
            continue
            
    clean_cells.append(cell)

# Step 2: Apply Enhancements to Clean Cells
new_cells = []
new_cells.append(hero_cell) # Always start with Hero

inserted_keys = set()

for cell in clean_cells:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if not source:
            new_cells.append(cell)
            continue
            
        first_line = source[0].strip()
        
        # Check specific distribution keys
        matched_dist = False
        for key, md_lines in dist_map.items():
            if key in first_line and key not in inserted_keys:
                # Special handling for Continuous Intro
                if "Uniform" in key:
                    new_cells.append(intro_continuous)
                
                # Special handling for Discrete Intro (Triggered by Bernoulli)
                if "Bernoulli" in key:
                    new_cells.append(intro_discrete)

                # Insert the specific markdown for this distribution
                new_cells.append(create_markdown_cell(md_lines))
                inserted_keys.add(key)
                matched_dist = True
                break
        
        new_cells.append(cell) # Add the code cell itself
        
    elif cell['cell_type'] == 'markdown':
        # Keep original markdown cells that weren't filtered out
        new_cells.append(cell)

nb['cells'] = new_cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook enhanced successfully with robust matching and idempotency!")
