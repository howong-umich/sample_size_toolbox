# Sample Size Calculator
A Streamlit web app for practical sample size planning.

This calculator supports both major sample size questions:
1. **Precision / margin of error**  
   Use this when you want to estimate a mean, proportion, or total with a chosen margin of error.

2. **Power**  
   Use this when you want enough sample to detect a meaningful effect with a chosen probability.

The app is designed for researchers, analysts, survey statisticians, public health teams, biomedical researchers, and data scientists who need transparent sample size calculations without repeatedly searching for formulas.

---

## Features

### Precision / margin-of-error calculations
Supported parameters:
- Mean
- Proportion
- Total

Supported designs:
- Simple random sampling
- Stratified sampling
  - Proportional allocation
  - Neyman allocation
- Equal-size cluster sampling

Additional options:
- Optional finite population correction
- Stratum-level allocation tables
- Design effect for clustered samples
- Achieved margin of error after rounding
- Warnings when sample size exceeds the finite population

---

### Power calculations
Supported tests/designs:
- One-sample mean
- One-sample proportion
- Paired mean
- Two independent means
- Two independent proportions
- Advanced custom Wald formula

Additional options:
- One-sided or two-sided tests
- User-specified alpha
- User-specified power
- Unequal allocation ratio for two independent means
- Optional design-effect adjustment
- Optional equal-size cluster design effect
- Optional approximate finite population correction

---

## Why this app exists
Sample size planning is often scattered across textbooks, survey manuals, clinical trial references, and online calculators. This app brings common planning calculations into one interactive tool.

The goal is not only to give a final number, but also to make the assumptions visible:

- What variance was assumed?
- What effect size was considered meaningful?
- Was clustering included?
- Was finite population correction used?
- How were strata allocated?
- What was the sample size before and after rounding?

A sample size number without its assumptions is not very useful. This app tries to make those assumptions explicit.

---

## Installation

### 1. Clone or download the project
Place "sample_size.py" in your project folder.

Example structure:

sample-size-calculator/
├── sample_size.py
├── requirements.txt
└── README.md

### 2. Create a virtual environment
On macOS or Linux:

bash
python -m venv .venv
source .venv/bin/activate

On Windows:

bash
python -m venv .venv
.venv\Scripts\activate

### 3. Install dependencies

Download "requirements.txt", then install 

pip install -r requirements.txt

Recommended Python version: Python 3.10 or later

---

## Running the app locally

From the project folder, run:

bash
streamlit run sample_size.py

Streamlit will open the app in your browser.

If it does not open automatically, copy the local URL shown in the terminal and paste it into your browser.

---

## How to use the app

### Step 1: Set global inputs

Use the sidebar to choose:
- Population size (N), if relevant
- Whether to use finite population correction
- Calculation mode:
  - Precision / margin of error
  - Power
- Whether to open calculation details by default

If population size is unknown or not relevant, leave N = 0.

---

## Precision mode
Use precision mode when your main question is:

> How many observations do I need to estimate something accurately enough?

Examples:
- Estimate unemployment within plus or minus 3 percentage points
- Estimate average income within a chosen margin of error
- Estimate vaccination coverage by province
- Estimate a population total

### Supported sampling designs

#### 1. Simple random sampling
Use this as the basic benchmark.

Required inputs usually include:
- Confidence level
- Margin of error
- Expected standard deviation for a mean or total
- Expected proportion for a proportion

If the expected proportion is unknown, p = 0.5 is often used because it gives the largest required sample size for a simple proportion.

#### 2. Stratified sampling
Use this when the population is divided into strata, such as:
- Region
- Sex
- Age group
- Urban/rural location
- Hospital type
- Disease stage

The app supports:
- Proportional allocation
- Neyman allocation

With finite population correction enabled, you must enter stratum population counts, and the counts must sum to the sidebar population size N.

Without finite population correction, you enter stratum weights, and the weights must sum to 1.

#### 3. Equal-size cluster sampling

Use this when sampling clusters of equal size, such as:
- Villages
- Schools
- Clinics
- Hospitals
- Households

Required inputs include:
- Expected variability
- Cluster size (m)
- Intracluster correlation (rho)

The app uses the common design effect approximation:

deff approximately equals to 1 + (m - 1)*rho

Cluster sampling often increases the required sample size because observations within the same cluster tend to be similar.

---

## Power mode

Use power mode when your main question is:

> How many observations do I need to detect a meaningful effect?

Examples:
- Detect a treatment effect in a clinical trial
- Detect a difference between two proportions
- Detect a mean difference between two groups
- Power an A/B test
- Plan a paired before-and-after study

Required inputs usually include:
- Significance level (alpha)
- Desired power (1 - beta)
- Test sidedness
- Expected standard deviation or expected proportions
- Minimum meaningful effect size
- Allocation ratio, if group sizes are unequal

The app uses standard large-sample normal approximation formulas.

---

## Interpreting the output

The most important section is:
"Final answer"

This gives the rounded sample size you should carry forward.

Depending on the design, the final answer may show:
- Required total sample size
- Required sample size per group
- Required number of pairs
- Required number of clusters
- Cluster-rounded number of elements
- Design effect

The app also provides expandable calculation details, including intermediate values such as:
- z values
- raw unrounded sample size
- design effect
- FPC-adjusted sample size
- stratum allocation
- achieved margin of error

---

## Finite population correction
Finite population correction is useful when sampling without replacement from a finite population, especially when the sample is a large fraction of the population.

The app uses the standard FPC-adjusted raw sample size:

n_fpc = (N * n_0)/(N + n_0 - 1)

where n_0 is the required sample size before FPC.

Notes:
- For SRS precision, FPC uses the sidebar population size N.
- For stratified precision, FPC uses stratum population counts.
- For equal-size cluster precision, FPC uses the number of population clusters.
- For power calculations, FPC is applied as an approximate unit-level adjustment after any design-effect adjustment.

For complex finite-population designs, especially clustered or weighted designs, a design-specific method or simulation may be preferable.

---

## Example use cases

### Example 1: Estimating a proportion

Question:

> How many people do I need to estimate a proportion with 95% confidence and a margin of error of 3 percentage points?

Use:
- Mode: Precision / margin of error
- Parameter: Proportion
- Design: SRS
- Confidence level: 0.95
- Margin of error: 0.03
- Expected proportion: 0.50

The app will return the required sample size and the achieved margin of error after rounding.

---

### Example 2: Stratified survey

Question:

> How many households should be sampled across regions if each region has a different population share and variability?

Use:
- Mode: Precision / margin of error
- Parameter: Mean, Proportion, or Total
- Design: Stratified
- Allocation method:
  - Proportional allocation, or
  - Neyman allocation

The app returns:
- Total sample size
- Final stratum sample sizes (n_h)
- Allocation shares
- Achieved margin of error

---

### Example 3: Cluster survey

Question:

> How many schools should be sampled if students within the same school are correlated?

Use:
- Mode: Precision / margin of error
- Design: Equal-size cluster
- Enter:
  - Cluster size (m)
  - Intracluster correlation (rho)
  - Expected variability

The app returns:
- Required number of clusters
- Approximate number of sampled elements
- Design effect

---

### Example 4: A/B test or two-group comparison

Question:

> How many users per group are needed to detect a difference between two conversion rates?

Use:
- Mode: Power
- Formula: Two independent proportions
- Enter:
  - Group 1 expected proportion
  - Group 2 expected proportion
  - Alpha
  - Desired power
  - Test sidedness

The app returns the required sample size for each group and the total sample size.

---

## Important assumptions

This calculator uses standard planning formulas. These formulas are useful, but they are still approximations.

Be especially careful when you have:
- Very rare binary outcomes
- Very small samples
- Few clusters
- Highly unequal cluster sizes
- Strong weighting effects
- Adaptive designs
- Longitudinal or multilevel designs
- Non-standard estimators
- Complex missing-data mechanisms

For complicated designs, consider simulation-based power or precision analysis.

---

## Nonresponse, dropout, and eligibility

The sample size returned by the app is the number of completed or analyzable observations needed.

If you expect nonresponse, dropout, ineligibility, or missing data, inflate the sample size manually.

For example, if the required completed sample size is 1,000 and the expected response rate is 80%, then the number to approach is:

1000 / 0.8 = 1250

If both eligibility and response rates matter, use:

\[
n_{\text{draw}} =
\frac{n_{\text{analysis}}}{\text{eligibility rate} \times \text{response rate}}
\]

#n_draw = n_analysis / (eligibility rate * response rate)

---

## Code overview

The app is contained in a single file:

sample_size.py

Main sections:
(i) General helpers
(ii) User guidance
(iii) Precision-based calculations
(iv) Power-based calculations
(v) Main Streamlit app

Important functions include:
- precision_mode()
- power_mode()
- z_alpha()
- z_power()
- fpc_adjusted_raw_n()
- finite_population_factor()
- largest_remainder_allocation()
- capped_largest_remainder_allocation()
- stratified_mean_variance()
- minimum_stratified_n_with_fpc()

---

## Limitations
This app is intended for planning and educational use.

It does not replace:
- A full sampling design review
- A biostatistical analysis plan
- Exact methods for small samples
- Specialized cluster-randomized trial software
- Simulation for complex designs
- Expert review for regulatory or high-stakes studies

The final sample size depends heavily on the assumptions entered by the user.

---

## Related resources
- Medium article: [Insert article link]
- Formula appendix: [Insert formula sheet link]
- Live app: [Insert Streamlit app link]
- Source code: [Insert GitHub link]

---

## License
```text
MIT License
```

---

## Author
Created by: Howard Wong

If you use this calculator in a report, protocol, article, or teaching material, please cite or link to the repository and accompanying formula appendix.
