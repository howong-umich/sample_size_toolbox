# sample_size.py

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


EPS = 1e-12
WEIGHT_TOL = 1e-6
NORM = NormalDist()


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def fail(message: str) -> None:
    st.error(message)
    st.stop()


def z_alpha(alpha: float, sided: str) -> float:
    if not 0 < alpha < 1:
        fail("Alpha must be between 0 and 1.")

    if sided == "Two-sided":
        return NORM.inv_cdf(1 - alpha / 2)

    return NORM.inv_cdf(1 - alpha)


def z_power(power: float) -> float:
    if not 0 < power < 1:
        fail("Power must be between 0 and 1.")

    return NORM.inv_cdf(power)


def ceil_sample_size(x: float) -> int:
    if not math.isfinite(x) or x < 0:
        fail("Computed sample size is invalid. Please check the inputs.")

    if x <= EPS:
        return 0

    return int(math.ceil(x))


def ceil_cap_to_population(x: float, N_pop: int) -> int:
    if N_pop < 0:
        fail("Population size cannot be negative.")

    return min(ceil_sample_size(x), int(N_pop))


def finite_population_factor(n: int, N_pop: int) -> float:
    """
    FPC variance multiplier using (N - n) / (N - 1).

    Returns 0 when n >= N.
    """
    if N_pop <= 0:
        fail("A positive population size is required for the finite population correction.")

    if n <= 0:
        return 1.0

    if n >= N_pop:
        return 0.0

    if N_pop == 1:
        return 0.0 if n >= 1 else 1.0

    return max(0.0, (N_pop - n) / (N_pop - 1))


def fpc_adjusted_raw_n(no_fpc_raw_n: float, N_pop: int) -> float:
    """
    Standard finite-population-corrected raw sample size:

        n_fpc = N n0 / (N + n0 - 1)

    where n0 is the no-FPC required sample size.
    """
    if N_pop <= 0:
        fail("A positive population size is required for the finite population correction.")

    if not math.isfinite(no_fpc_raw_n) or no_fpc_raw_n < 0:
        fail("Computed no-FPC sample size is invalid.")

    if no_fpc_raw_n <= EPS:
        return 0.0

    denominator = N_pop + no_fpc_raw_n - 1

    if denominator <= EPS:
        return float(N_pop)

    return (N_pop * no_fpc_raw_n) / denominator


def format_metric_value(value: int | float | str) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"

    if isinstance(value, (float, np.floating)):
        value_float = float(value)

        if math.isnan(value_float):
            return "NA"

        if math.isinf(value_float):
            return "∞"

        return f"{value_float:,.4f}"

    return str(value)


def show_final_answer(
    message: str,
    metrics: list[tuple[str, int | float | str]],
) -> None:
    st.subheader("Final answer")
    st.success(message)

    cols = st.columns(min(len(metrics), 3))

    for idx, (label, value) in enumerate(metrics):
        cols[idx % len(cols)].metric(label, format_metric_value(value))


def show_detail_table(
    title: str,
    df: pd.DataFrame,
    show_details: bool,
) -> None:
    with st.expander(title, expanded=show_details):
        st.dataframe(df, hide_index=True, use_container_width=True)


def largest_remainder_allocation(total_n: int, shares: Iterable[float]) -> list[int]:
    if total_n < 0:
        fail("Total sample size for allocation cannot be negative.")

    share_array = np.asarray(list(shares), dtype=float)

    if len(share_array) == 0:
        fail("No allocation shares were provided.")

    if np.any(~np.isfinite(share_array)):
        fail("Allocation shares contain invalid values.")

    if np.any(share_array < 0):
        fail("Allocation shares cannot be negative.")

    share_sum = float(share_array.sum())

    if share_sum <= EPS:
        fail("Allocation shares sum to zero.")

    share_array = share_array / share_sum

    raw = share_array * total_n
    base = np.floor(raw).astype(int)
    remainder = int(total_n - base.sum())

    if remainder > 0:
        fractional = raw - base
        order = np.argsort(-fractional)

        for idx in order[:remainder]:
            base[idx] += 1

    return base.tolist()


def capped_largest_remainder_allocation(
    total_n: int,
    shares: Iterable[float],
    caps: Iterable[int],
) -> list[int]:
    """
    Allocate total_n according to shares while not exceeding integer caps.

    This is used for finite populations so no stratum receives more sampled units
    than actually exist in that stratum.
    """
    total_n = int(total_n)
    share_array = np.asarray(list(shares), dtype=float)
    caps_array = np.asarray(list(caps), dtype=int)

    if total_n < 0:
        fail("Total sample size for allocation cannot be negative.")

    if len(share_array) == 0:
        fail("No allocation shares were provided.")

    if len(share_array) != len(caps_array):
        fail("Allocation shares and caps must have the same length.")

    if np.any(~np.isfinite(share_array)):
        fail("Allocation shares contain invalid values.")

    if np.any(share_array < 0):
        fail("Allocation shares cannot be negative.")

    if np.any(caps_array < 0):
        fail("Allocation caps cannot be negative.")

    cap_sum = int(caps_array.sum())

    if total_n > cap_sum:
        fail(
            f"Cannot allocate {total_n:,} sampled units because the finite population "
            f"contains only {cap_sum:,} units across the allocation cells."
        )

    if total_n == 0:
        return [0] * len(share_array)

    allocation = np.zeros(len(share_array), dtype=int)
    remaining = total_n

    while remaining > 0:
        available = caps_array - allocation
        active = (available > 0) & (share_array > 0)

        if not np.any(active):
            active = available > 0
            active_shares = np.where(active, 1.0, 0.0)
        else:
            active_shares = np.where(active, share_array, 0.0)

        active_sum = float(active_shares.sum())

        if active_sum <= EPS:
            fail("No feasible allocation cells remain.")

        normalized = active_shares / active_sum
        raw = normalized * remaining

        add = np.floor(raw).astype(int)
        add = np.minimum(add, available)

        progress = int(add.sum())

        if progress > 0:
            allocation += add
            remaining -= progress
            continue

        fractional = raw - np.floor(raw)

        order = sorted(
            np.where(active)[0].tolist(),
            key=lambda i: (fractional[i], share_array[i], available[i]),
            reverse=True,
        )

        for idx in order:
            if remaining <= 0:
                break

            if allocation[idx] < caps_array[idx]:
                allocation[idx] += 1
                remaining -= 1
                progress += 1

        if progress <= 0:
            fail("Unable to complete the capped allocation.")

    return allocation.tolist()


def apply_population_warning(
    sample_n: int,
    N_pop: int,
    context: str = "sample size",
) -> None:
    if N_pop > 0 and sample_n > N_pop:
        st.warning(
            f"The computed {context} is {sample_n:,}, which is larger than the "
            f"specified population size N = {N_pop:,}.\n\n"
            "Do not sample more unique units than exist in the population. "
            "This calculator does not apply the finite population correction for this "
            "specific result.\n\n"
            "Practical options are:\n\n"
            "1. Treat the study as a census and sample all available units.\n"
            "2. Use the finite population correction.\n"
            "3. Accept a larger margin of error, lower confidence level, or lower power.\n"
            "4. Revisit the assumed variance, ICC, design effect, or target effect size.\n"
            "5. If sampling all N units, report the achieved precision or power for n = N."
        )


def validate_weights_sum_to_one(weights: np.ndarray) -> None:
    if np.any(~np.isfinite(weights)):
        fail("Stratum weights contain invalid values.")

    weight_sum = float(weights.sum())

    if abs(weight_sum - 1.0) > WEIGHT_TOL:
        fail(
            f"Stratum weights must sum to 1. Current sum is {weight_sum:.8f}. "
            "Please revise the weights before continuing."
        )


def validate_positive_population_for_fpc(N_pop: int, context: str = "this calculation") -> None:
    if N_pop <= 0:
        fail(
            f"A positive population size N is required to use the finite population "
            f"correction for {context}. Please enter N in the sidebar."
        )


def achieved_margin_from_mean_variance(
    mean_variance: float,
    z: float,
    parameter: str,
    N_pop: int,
) -> float:
    if not math.isfinite(mean_variance):
        return math.inf

    scale = float(N_pop) if parameter == "Total" else 1.0

    return z * math.sqrt(max(mean_variance, 0.0)) * scale


def stratified_mean_variance(
    weights: np.ndarray,
    s_h: np.ndarray,
    allocations: Iterable[int],
    use_fpc: bool,
    counts: np.ndarray | None = None,
) -> float:
    allocations_array = np.asarray(list(allocations), dtype=int)

    if len(weights) != len(s_h) or len(weights) != len(allocations_array):
        fail("Stratified variance inputs have incompatible lengths.")

    if use_fpc:
        if counts is None:
            fail("Stratum population counts are required for FPC stratified variance.")

        counts_array = np.asarray(counts, dtype=int)

        if len(counts_array) != len(weights):
            fail("Stratum population counts have incompatible length.")
    else:
        counts_array = None

    total_variance = 0.0

    for idx, (weight, sd, n_h) in enumerate(zip(weights, s_h, allocations_array)):
        if weight <= EPS or sd <= EPS:
            continue

        if n_h <= 0:
            return math.inf

        if use_fpc:
            assert counts_array is not None
            N_h = int(counts_array[idx])

            if N_h <= 1:
                if n_h >= N_h:
                    continue

                return math.inf

            fpc = finite_population_factor(int(n_h), N_h)
            total_variance += (weight ** 2) * (sd ** 2) * fpc / n_h

        else:
            total_variance += (weight ** 2) * (sd ** 2) / n_h

    return float(total_variance)


def stratified_fpc_raw_total_n(
    weights: np.ndarray,
    s_h: np.ndarray,
    allocation_shares: np.ndarray,
    counts: np.ndarray,
    target_variance: float,
) -> float:
    """
    Continuous-allocation approximation for stratified sample size with
    stratum-level FPC.

    With n_h = a_h n,

        Var = A / n - B

    so

        n >= A / (target_variance + B)
    """
    if target_variance <= EPS:
        fail("Target variance is too small.")

    A = 0.0
    B = 0.0

    for weight, sd, share, N_h in zip(weights, s_h, allocation_shares, counts):
        if weight <= EPS or sd <= EPS:
            continue

        if N_h <= 1:
            fail(
                "A stratum with population count 1 cannot have positive within-stratum "
                "variability when using FPC. Set its variability to 0 or revise the strata."
            )

        if share <= EPS:
            fail(
                "A stratum with positive variability has zero allocation share. "
                "Please revise the allocation method or stratum inputs."
            )

        A += (weight ** 2) * (sd ** 2) * N_h / (share * (N_h - 1))
        B += (weight ** 2) * (sd ** 2) / (N_h - 1)

    if A <= EPS:
        return 0.0

    return A / (target_variance + B)


def minimum_stratified_n_with_fpc(
    initial_raw_n: float,
    target_variance: float,
    weights: np.ndarray,
    s_h: np.ndarray,
    allocation_shares: np.ndarray,
    counts: np.ndarray,
) -> tuple[int, list[int], float]:
    total_population = int(counts.sum())

    if total_population <= 0:
        fail("Total finite population across strata must be positive.")

    start_n = ceil_cap_to_population(initial_raw_n, total_population)

    def allocation_and_variance(n_total: int) -> tuple[list[int], float]:
        allocation = capped_largest_remainder_allocation(
            n_total,
            allocation_shares,
            counts,
        )
        variance = stratified_mean_variance(
            weights,
            s_h,
            allocation,
            use_fpc=True,
            counts=counts,
        )
        return allocation, variance

    start_allocation, start_variance = allocation_and_variance(start_n)

    if start_variance <= target_variance:
        return start_n, start_allocation, start_variance

    if start_n >= total_population:
        fail(
            "Even a census of all finite-population units did not satisfy the target "
            "variance. Please check the inputs."
        )

    hi = max(start_n + 1, 1)
    hi = min(hi, total_population)

    while hi < total_population:
        _, hi_variance = allocation_and_variance(hi)

        if hi_variance <= target_variance:
            break

        hi = min(total_population, max(hi + 1, hi * 2))

    hi_allocation, hi_variance = allocation_and_variance(hi)

    if hi_variance > target_variance and hi >= total_population:
        fail(
            "Even a census of all finite-population units did not satisfy the target "
            "variance. Please check the inputs."
        )

    left = start_n + 1
    right = hi
    best_n = hi
    best_allocation = hi_allocation
    best_variance = hi_variance

    while left <= right:
        mid = (left + right) // 2
        mid_allocation, mid_variance = allocation_and_variance(mid)

        if mid_variance <= target_variance:
            best_n = mid
            best_allocation = mid_allocation
            best_variance = mid_variance
            right = mid - 1
        else:
            left = mid + 1

    return best_n, best_allocation, best_variance


# ---------------------------------------------------------------------
# User guidance
# ---------------------------------------------------------------------

def show_precision_guidance() -> None:
    with st.expander(
        "Quick guide: what inputs do I need for a precision calculation?",
        expanded=True,
    ):
        st.markdown(
            """
Use this mode when the goal is to estimate a **mean**, **proportion**, or **total**
with a desired margin of error.

You will usually need:

- **Confidence level**: commonly 0.95.
- **Margin of error**: the largest acceptable error in the estimate.
- **Expected variability**:
  - For a mean or total: an anticipated standard deviation.
  - For a proportion: an anticipated proportion.
  - For strata: stratum population weights or counts and stratum-specific SDs or proportions.
  - For clusters: cluster size and intracluster correlation.
- **Population size**, if using the finite population correction.

These inputs often come from:

- prior studies,
- past surveys,
- pilot studies,
- administrative data,
- subject-matter expertise,
- or conservative planning assumptions.

If the anticipated proportion is unknown, 0.50 is often used because it gives the
largest required sample size for a simple proportion.

The finite population correction is useful when the sample is expected to be a
large fraction of the total finite population.
            """
        )


def show_power_guidance() -> None:
    with st.expander(
        "Quick guide: what inputs do I need for a power calculation?",
        expanded=True,
    ):
        st.markdown(
            """
Use this mode when the goal is to detect a specified effect with a desired probability,
called **power**.

You will usually need:

- **Significance level alpha**: commonly 0.05.
- **Desired power**: commonly 0.80 or 0.90.
- **Effect size to detect**:
  - mean difference,
  - proportion difference,
  - paired difference,
  - or another planned effect.
- **Expected variability**:
  - standard deviation for mean outcomes,
  - expected proportions for binary outcomes,
  - standard deviation of paired differences for paired designs.
- **Allocation ratio**, if the two groups will not have equal sample sizes.
- **Population size**, if applying an approximate finite population correction.

These inputs often come from prior studies, past surveys, pilot data,
administrative records, clinical or policy relevance, or expert judgment.

The final rounded sample size shown in the **Final answer** section is the main
number users should carry forward. More technical intermediate quantities are
shown only under **Calculation details**.
            """
        )


# ---------------------------------------------------------------------
# Precision-based calculations
# ---------------------------------------------------------------------

def precision_mode(N_pop: int, use_fpc: bool, show_details: bool) -> None:
    st.header("Precision / margin-of-error calculation")

    if use_fpc:
        st.info(
            "Finite population correction is enabled. For SRS and stratified sampling, "
            "enter a positive population size N in the sidebar. For equal-size cluster "
            "sampling, the app also asks for the number of population clusters."
        )
    else:
        st.info(
            "Use this mode when you want an estimate to be within a chosen margin of error. "
            "Finite population correction is currently turned off."
        )

    show_precision_guidance()

    st.subheader("Step 1. Choose what you are estimating")

    parameter = st.selectbox(
        "Parameter to estimate",
        ["Mean", "Proportion", "Total"],
        help=(
            "Choose Mean for an average value, Proportion for a percentage or rate, "
            "or Total for a population total."
        ),
    )

    design = st.selectbox(
        "Sampling design",
        ["SRS", "Stratified", "Equal-size cluster"],
        help=(
            "SRS is simple random sampling. Stratified sampling divides the population "
            "into strata. Equal-size cluster sampling samples clusters of the same size."
        ),
    )

    if use_fpc and design in ["SRS", "Stratified"]:
        validate_positive_population_for_fpc(N_pop, context=f"{design} precision")

    st.subheader("Step 2. Choose the precision target")

    col1, col2 = st.columns(2)

    with col1:
        confidence = st.number_input(
            "Confidence level",
            min_value=0.500,
            max_value=0.999,
            value=0.950,
            step=0.010,
            format="%.3f",
            help="Common choices are 0.90, 0.95, and 0.99.",
        )

    with col2:
        margin_label = (
            "Allowed margin of error for the total"
            if parameter == "Total"
            else "Allowed margin of error"
        )

        margin_error = st.number_input(
            margin_label,
            min_value=1e-12,
            value=0.05,
            step=0.01,
            format="%.6f",
            help=(
                "This is the largest acceptable error. For example, use 0.03 if a "
                "proportion estimate should be within plus or minus 3 percentage points."
            ),
        )

    alpha = 1 - confidence
    z = NORM.inv_cdf(1 - alpha / 2)

    if parameter == "Total":
        if N_pop <= 0:
            fail(
                "For a total, please enter a positive population size N in the sidebar. "
                "The total-scale margin of error is converted to the mean scale using N."
            )

        target_se = margin_error / (z * N_pop)
    else:
        target_se = margin_error / z

    target_variance = target_se ** 2

    if target_variance <= EPS:
        fail("Target variance is too small. Please check the margin of error.")

    # -----------------------------------------------------------------
    # SRS
    # -----------------------------------------------------------------

    if design == "SRS":
        st.subheader("Step 3. Enter expected variability")

        if parameter in ["Mean", "Total"]:
            st.info(
                "Required input: an anticipated unit-level standard deviation. "
                "This can come from a previous survey, pilot study, administrative data, "
                "or a reasonable planning assumption."
            )

            sigma = st.number_input(
                "Expected standard deviation",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.6f",
            )

            unit_variance = sigma ** 2

        else:
            st.info(
                "Required input: an anticipated proportion. If unknown, 0.50 is a "
                "conservative choice because it usually gives the largest sample size."
            )

            p = st.number_input(
                "Expected proportion",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.6f",
            )

            unit_variance = p * (1 - p)

        no_fpc_raw_n = unit_variance / target_variance

        if use_fpc:
            n_raw = fpc_adjusted_raw_n(no_fpc_raw_n, N_pop)
            n_final = ceil_cap_to_population(n_raw, N_pop)

            if unit_variance <= EPS:
                achieved_mean_variance = 0.0
            elif n_final <= 0:
                achieved_mean_variance = math.inf
            else:
                achieved_mean_variance = (
                    unit_variance
                    * finite_population_factor(n_final, N_pop)
                    / n_final
                )
        else:
            n_raw = no_fpc_raw_n
            n_final = ceil_sample_size(n_raw)

            if unit_variance <= EPS:
                achieved_mean_variance = 0.0
            elif n_final <= 0:
                achieved_mean_variance = math.inf
            else:
                achieved_mean_variance = unit_variance / n_final

        achieved_margin = achieved_margin_from_mean_variance(
            achieved_mean_variance,
            z,
            parameter,
            N_pop,
        )

        fpc_phrase = " with FPC" if use_fpc else ""
        show_final_answer(
            f"Use **{n_final:,} sampled units**{fpc_phrase}. This rounded value is "
            "the main sample size to carry forward for this SRS calculation.",
            [("Required sample size n", n_final)],
        )

        detail_rows = [
            {"Quantity": "Confidence level", "Value": confidence},
            {"Quantity": "Margin of error", "Value": margin_error},
            {"Quantity": "z value", "Value": z},
            {"Quantity": "Target standard error", "Value": target_se},
            {"Quantity": "Unit-level variance used", "Value": unit_variance},
            {"Quantity": "Raw no-FPC sample size", "Value": no_fpc_raw_n},
            {"Quantity": "FPC used", "Value": "Yes" if use_fpc else "No"},
        ]

        if use_fpc:
            detail_rows.extend(
                [
                    {"Quantity": "Population size N", "Value": N_pop},
                    {"Quantity": "Raw FPC-adjusted sample size", "Value": n_raw},
                    {
                        "Quantity": "FPC factor at final n",
                        "Value": finite_population_factor(n_final, N_pop),
                    },
                ]
            )

        detail_rows.extend(
            [
                {"Quantity": "Final rounded sample size", "Value": n_final},
                {"Quantity": "Achieved margin of error", "Value": achieved_margin},
            ]
        )

        detail_df = pd.DataFrame(detail_rows)
        show_detail_table("Calculation details", detail_df, show_details)

        if not use_fpc:
            apply_population_warning(n_final, N_pop)

    # -----------------------------------------------------------------
    # Stratified sampling
    # -----------------------------------------------------------------

    elif design == "Stratified":
        st.subheader("Step 3. Enter stratum information")

        if use_fpc:
            st.info(
                "Required inputs: one row per stratum, the finite population count "
                "for each stratum, and the expected variability within each stratum. "
                "The stratum counts must sum to the sidebar population size N."
            )
        else:
            st.info(
                "Required inputs: one row per stratum, the population weight for each "
                "stratum, and the expected variability within each stratum. The stratum "
                "weights must sum to 1. For example, if 40% of the population is in a "
                "stratum, enter 0.40."
            )

        allocation_method = st.radio(
            "Allocation method",
            ["Proportional allocation", "Neyman allocation"],
            horizontal=True,
            help=(
                "Proportional allocation samples in proportion to stratum size. "
                "Neyman allocation samples more heavily from larger and more variable strata."
            ),
        )

        H = int(
            st.number_input(
                "Number of strata",
                min_value=2,
                value=3,
                step=1,
            )
        )

        if use_fpc and H > N_pop:
            fail(
                "With FPC enabled, the number of strata cannot exceed the population "
                "size if every stratum must have a positive population count."
            )

        default_weights = [1 / H] * H

        if use_fpc:
            default_counts = largest_remainder_allocation(N_pop, default_weights)

            if parameter in ["Mean", "Total"]:
                default_df = pd.DataFrame(
                    {
                        "Stratum": [f"Stratum {i + 1}" for i in range(H)],
                        "Population count (N_h)": default_counts,
                        "SD (S_h)": [1.0] * H,
                    }
                )

                column_config = {
                    "Stratum": st.column_config.TextColumn("Stratum"),
                    "Population count (N_h)": st.column_config.NumberColumn(
                        "Population count (N_h)",
                        min_value=1,
                        step=1,
                        format="%d",
                    ),
                    "SD (S_h)": st.column_config.NumberColumn(
                        "SD (S_h)",
                        min_value=0.0,
                        step=0.01,
                        format="%.6f",
                    ),
                }

            else:
                default_df = pd.DataFrame(
                    {
                        "Stratum": [f"Stratum {i + 1}" for i in range(H)],
                        "Population count (N_h)": default_counts,
                        "Proportion (p_h)": [0.50] * H,
                    }
                )

                column_config = {
                    "Stratum": st.column_config.TextColumn("Stratum"),
                    "Population count (N_h)": st.column_config.NumberColumn(
                        "Population count (N_h)",
                        min_value=1,
                        step=1,
                        format="%d",
                    ),
                    "Proportion (p_h)": st.column_config.NumberColumn(
                        "Proportion (p_h)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        format="%.6f",
                    ),
                }

        else:
            if parameter in ["Mean", "Total"]:
                default_df = pd.DataFrame(
                    {
                        "Stratum": [f"Stratum {i + 1}" for i in range(H)],
                        "Weight (W_h)": default_weights,
                        "SD (S_h)": [1.0] * H,
                    }
                )

                column_config = {
                    "Stratum": st.column_config.TextColumn("Stratum"),
                    "Weight (W_h)": st.column_config.NumberColumn(
                        "Weight (W_h)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.001,
                        format="%.6f",
                    ),
                    "SD (S_h)": st.column_config.NumberColumn(
                        "SD (S_h)",
                        min_value=0.0,
                        step=0.01,
                        format="%.6f",
                    ),
                }

            else:
                default_df = pd.DataFrame(
                    {
                        "Stratum": [f"Stratum {i + 1}" for i in range(H)],
                        "Weight (W_h)": default_weights,
                        "Proportion (p_h)": [0.50] * H,
                    }
                )

                column_config = {
                    "Stratum": st.column_config.TextColumn("Stratum"),
                    "Weight (W_h)": st.column_config.NumberColumn(
                        "Weight (W_h)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.001,
                        format="%.6f",
                    ),
                    "Proportion (p_h)": st.column_config.NumberColumn(
                        "Proportion (p_h)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        format="%.6f",
                    ),
                }

        edited_df = st.data_editor(
            default_df,
            hide_index=True,
            num_rows="fixed",
            column_config=column_config,
            use_container_width=True,
            key=f"strata_editor_{parameter}_{H}_{N_pop}_{'fpc' if use_fpc else 'nofpc'}",
        )

        df = edited_df.copy()

        if use_fpc:
            df["Population count (N_h)"] = pd.to_numeric(
                df["Population count (N_h)"],
                errors="coerce",
            )

            if df["Population count (N_h)"].isna().any():
                fail("All stratum population counts must be numeric.")

            counts_float = df["Population count (N_h)"].to_numpy(dtype=float)

            if np.any(np.abs(counts_float - np.round(counts_float)) > WEIGHT_TOL):
                fail("Stratum population counts must be whole numbers.")

            counts = np.round(counts_float).astype(int)

            if np.any(counts <= 0):
                fail("All stratum population counts must be positive.")

            if int(counts.sum()) != N_pop:
                fail(
                    f"Stratum population counts must sum to sidebar N = {N_pop:,}. "
                    f"Current sum is {int(counts.sum()):,}."
                )

            weights = counts / counts.sum()
            df["Weight (W_h)"] = weights

        else:
            df["Weight (W_h)"] = pd.to_numeric(df["Weight (W_h)"], errors="coerce")

            if df["Weight (W_h)"].isna().any():
                fail("All stratum weights must be numeric.")

            weights = df["Weight (W_h)"].to_numpy(dtype=float)

            if np.any(weights < 0):
                fail("Stratum weights cannot be negative.")

            validate_weights_sum_to_one(weights)
            counts = None

        if parameter in ["Mean", "Total"]:
            df["SD (S_h)"] = pd.to_numeric(df["SD (S_h)"], errors="coerce")

            if df["SD (S_h)"].isna().any():
                fail("All stratum standard deviations must be numeric.")

            s_h = df["SD (S_h)"].to_numpy(dtype=float)

            if np.any(s_h < 0):
                fail("Stratum standard deviations cannot be negative.")

        else:
            df["Proportion (p_h)"] = pd.to_numeric(
                df["Proportion (p_h)"],
                errors="coerce",
            )

            if df["Proportion (p_h)"].isna().any():
                fail("All stratum proportions must be numeric.")

            p_h = df["Proportion (p_h)"].to_numpy(dtype=float)

            if np.any((p_h < 0) | (p_h > 1)):
                fail("All stratum proportions must be between 0 and 1.")

            s_h = np.sqrt(p_h * (1 - p_h))

        if use_fpc:
            assert counts is not None

            if np.any((counts <= 1) & (s_h > EPS)):
                fail(
                    "A stratum with population count 1 cannot have positive "
                    "within-stratum variability when FPC is used. Set that stratum's "
                    "SD to 0, set its proportion to 0 or 1, or revise the strata."
                )

        if allocation_method == "Proportional allocation":
            allocation_shares = weights.copy()
            effective_variance = float(np.sum(weights * s_h ** 2))

        else:
            neyman_denom = float(np.sum(weights * s_h))

            if neyman_denom <= EPS:
                fail(
                    "Neyman allocation is undefined because all W_h S_h values are zero. "
                    "Please check the stratum variances or proportions."
                )

            allocation_shares = weights * s_h / neyman_denom
            effective_variance = neyman_denom ** 2

        if use_fpc:
            assert counts is not None

            n_raw = stratified_fpc_raw_total_n(
                weights=weights,
                s_h=s_h,
                allocation_shares=allocation_shares,
                counts=counts,
                target_variance=target_variance,
            )

            n_final, final_allocations, achieved_mean_variance = minimum_stratified_n_with_fpc(
                initial_raw_n=n_raw,
                target_variance=target_variance,
                weights=weights,
                s_h=s_h,
                allocation_shares=allocation_shares,
                counts=counts,
            )

        else:
            n_raw = effective_variance / target_variance
            n_final = ceil_sample_size(n_raw)

            if n_final > 0:
                final_allocations = largest_remainder_allocation(
                    n_final,
                    allocation_shares,
                )
            else:
                final_allocations = [0] * len(allocation_shares)

            achieved_mean_variance = stratified_mean_variance(
                weights=weights,
                s_h=s_h,
                allocations=final_allocations,
                use_fpc=False,
            )

        achieved_margin = achieved_margin_from_mean_variance(
            achieved_mean_variance,
            z,
            parameter,
            N_pop,
        )

        fpc_phrase = " with FPC" if use_fpc else ""
        show_final_answer(
            f"Use **{n_final:,} total sampled units**{fpc_phrase}. For a stratified "
            "design, the required figures to carry forward are the total sample size "
            "and the **Final n_h to sample** column below.",
            [("Required total sample size n", n_final)],
        )

        allocation_display = df.copy()
        allocation_display["Within-stratum SD used"] = s_h
        allocation_display["Allocation share"] = allocation_shares
        allocation_display["Final n_h to sample"] = final_allocations

        if use_fpc:
            assert counts is not None
            allocation_display["Stratum sampling fraction"] = (
                np.asarray(final_allocations) / counts
            )
            allocation_display["Stratum FPC factor"] = [
                finite_population_factor(int(n_h), int(N_h))
                for n_h, N_h in zip(final_allocations, counts)
            ]

        st.subheader("Required stratum allocation")
        st.dataframe(allocation_display, hide_index=True, use_container_width=True)

        if allocation_method == "Neyman allocation":
            st.caption(
                "Neyman allocation assigns larger sample shares to strata with larger "
                "population weights and larger within-stratum standard deviations."
            )

        if any(x == 0 for x in final_allocations) and n_final > 0:
            st.warning(
                "At least one stratum received zero sampled units after rounding. "
                "If every stratum must be represented, increase the total sample size "
                "or impose a minimum per-stratum allocation."
            )

        summary_rows = [
            {"Quantity": "Confidence level", "Value": confidence},
            {"Quantity": "Margin of error", "Value": margin_error},
            {"Quantity": "z value", "Value": z},
            {"Quantity": "Target standard error", "Value": target_se},
            {"Quantity": "FPC used", "Value": "Yes" if use_fpc else "No"},
            {"Quantity": "Allocation method", "Value": allocation_method},
        ]

        if use_fpc:
            summary_rows.extend(
                [
                    {"Quantity": "Population size N", "Value": N_pop},
                    {"Quantity": "Continuous-allocation FPC total n", "Value": n_raw},
                ]
            )
        else:
            summary_rows.extend(
                [
                    {"Quantity": "Effective variance factor", "Value": effective_variance},
                    {"Quantity": "Raw no-FPC total sample size", "Value": n_raw},
                ]
            )

        summary_rows.extend(
            [
                {"Quantity": "Final rounded total sample size", "Value": n_final},
                {
                    "Quantity": "Sum of final stratum sample sizes",
                    "Value": int(sum(final_allocations)),
                },
                {"Quantity": "Achieved margin of error", "Value": achieved_margin},
            ]
        )

        summary_df = pd.DataFrame(summary_rows)

        detail_df = allocation_display.copy()
        detail_df["Raw n_h before rounding"] = allocation_shares * n_final

        show_detail_table("Calculation details: summary", summary_df, show_details)
        show_detail_table("Calculation details: stratum-level values", detail_df, show_details)

        if not use_fpc:
            apply_population_warning(n_final, N_pop)

    # -----------------------------------------------------------------
    # Equal-size cluster sampling
    # -----------------------------------------------------------------

    else:
        st.subheader("Step 3. Enter cluster information")

        st.info(
            "Required inputs: expected outcome variability, equal cluster size, and "
            "intracluster correlation. Cluster size may come from the planned field design. "
            "The intracluster correlation may come from prior studies, past surveys, "
            "pilot data, or a conservative planning assumption."
        )

        if parameter in ["Mean", "Total"]:
            sigma = st.number_input(
                "Expected standard deviation",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.6f",
            )

            unit_variance = sigma ** 2

        else:
            p = st.number_input(
                "Expected proportion",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.6f",
            )

            unit_variance = p * (1 - p)

        m = int(
            st.number_input(
                "Equal cluster size m",
                min_value=1,
                value=10,
                step=1,
            )
        )

        rho = st.number_input(
            "Intracluster correlation rho",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            format="%.6f",
        )

        deff = 1 + (m - 1) * rho

        M_clusters: int | None = None
        cluster_population_elements: int | None = None

        if use_fpc:
            default_M = max(1, math.ceil(N_pop / m)) if N_pop > 0 else 100

            M_clusters = int(
                st.number_input(
                    "Number of clusters in the population M",
                    min_value=1,
                    value=default_M,
                    step=1,
                    help=(
                        "For equal-size one-stage cluster sampling, the cluster-level "
                        "FPC uses the number of population clusters M."
                    ),
                )
            )

            cluster_population_elements = M_clusters * m

            st.caption(
                f"Implied element population under equal cluster size: "
                f"M × m = {M_clusters:,} × {m:,} = {cluster_population_elements:,}."
            )

            if N_pop > 0 and cluster_population_elements != N_pop:
                if parameter == "Total":
                    fail(
                        "For a total with equal-size cluster FPC, the sidebar population "
                        f"size N must equal M × m. Current sidebar N = {N_pop:,}, but "
                        f"M × m = {cluster_population_elements:,}."
                    )

                st.warning(
                    f"Sidebar N = {N_pop:,}, but M × m = {cluster_population_elements:,}. "
                    "For mean or proportion precision this does not affect the cluster-level "
                    "FPC calculation, but you may want these values to agree."
                )

        no_fpc_raw_elements = deff * unit_variance / target_variance
        no_fpc_raw_clusters = no_fpc_raw_elements / m

        if use_fpc:
            assert M_clusters is not None

            if M_clusters == 1:
                raw_clusters = 0.0 if unit_variance <= EPS else 1.0
            else:
                A = deff * unit_variance / m

                if A <= EPS:
                    raw_clusters = 0.0
                else:
                    raw_clusters = (A * M_clusters) / (
                        A + target_variance * (M_clusters - 1)
                    )

            n_clusters = ceil_cap_to_population(raw_clusters, M_clusters)
            cluster_rounded_elements = n_clusters * m

            if unit_variance <= EPS:
                achieved_mean_variance = 0.0
            elif n_clusters <= 0:
                achieved_mean_variance = math.inf
            else:
                A = deff * unit_variance / m
                achieved_mean_variance = (
                    A
                    * finite_population_factor(n_clusters, M_clusters)
                    / n_clusters
                )

        else:
            raw_clusters = no_fpc_raw_clusters
            n_clusters = ceil_sample_size(raw_clusters)
            cluster_rounded_elements = n_clusters * m

            if unit_variance <= EPS:
                achieved_mean_variance = 0.0
            elif cluster_rounded_elements <= 0:
                achieved_mean_variance = math.inf
            else:
                achieved_mean_variance = (
                    deff * unit_variance / cluster_rounded_elements
                )

        achieved_margin = achieved_margin_from_mean_variance(
            achieved_mean_variance,
            z,
            parameter,
            N_pop,
        )

        fpc_phrase = " with cluster-level FPC" if use_fpc else ""
        show_final_answer(
            f"Use **{n_clusters:,} clusters**{fpc_phrase}. With cluster size {m:,}, "
            f"this gives approximately **{cluster_rounded_elements:,} sampled elements**. "
            "If whole clusters must be sampled, the number of clusters is usually the "
            "main figure to carry forward.",
            [
                ("Required clusters", n_clusters),
                ("Cluster-rounded elements", cluster_rounded_elements),
                ("Design effect", deff),
            ],
        )

        detail_rows = [
            {"Quantity": "Confidence level", "Value": confidence},
            {"Quantity": "Margin of error", "Value": margin_error},
            {"Quantity": "z value", "Value": z},
            {"Quantity": "Target standard error", "Value": target_se},
            {"Quantity": "Unit-level variance used", "Value": unit_variance},
            {"Quantity": "Equal cluster size m", "Value": m},
            {"Quantity": "Intracluster correlation rho", "Value": rho},
            {"Quantity": "Design effect", "Value": deff},
            {"Quantity": "Raw no-FPC sampled elements", "Value": no_fpc_raw_elements},
            {"Quantity": "Raw no-FPC clusters", "Value": no_fpc_raw_clusters},
            {"Quantity": "FPC used", "Value": "Yes" if use_fpc else "No"},
        ]

        if use_fpc:
            assert M_clusters is not None
            detail_rows.extend(
                [
                    {"Quantity": "Population clusters M", "Value": M_clusters},
                    {
                        "Quantity": "Implied element population M × m",
                        "Value": cluster_population_elements,
                    },
                    {"Quantity": "Raw FPC-adjusted clusters", "Value": raw_clusters},
                    {
                        "Quantity": "Cluster-level FPC factor at final clusters",
                        "Value": finite_population_factor(n_clusters, M_clusters),
                    },
                ]
            )

        detail_rows.extend(
            [
                {"Quantity": "Final number of clusters", "Value": n_clusters},
                {
                    "Quantity": "Approximate elements after cluster rounding",
                    "Value": cluster_rounded_elements,
                },
                {"Quantity": "Achieved margin of error", "Value": achieved_margin},
            ]
        )

        detail_df = pd.DataFrame(detail_rows)

        show_detail_table("Calculation details", detail_df, show_details)

        if not use_fpc:
            apply_population_warning(
                cluster_rounded_elements,
                N_pop,
                context="cluster-rounded sample size",
            )


# ---------------------------------------------------------------------
# Power-based calculations
# ---------------------------------------------------------------------

def design_effect_widget() -> tuple[float, int | None]:
    with st.expander("Optional: design-effect adjustment", expanded=False):
        st.markdown(
            """
Most users should leave this as **None** unless the planned sample uses clustering
or another design feature that inflates the required sample size.
            """
        )

        adjustment = st.selectbox(
            "Design-effect adjustment",
            ["None", "Manual design effect", "Equal-size cluster design effect"],
        )

        if adjustment == "None":
            return 1.0, None

        if adjustment == "Manual design effect":
            deff = st.number_input(
                "Manual design effect",
                min_value=1.0,
                value=1.0,
                step=0.1,
                format="%.6f",
            )

            return float(deff), None

        m = int(
            st.number_input(
                "Equal cluster size m",
                min_value=1,
                value=10,
                step=1,
                key="power_cluster_m",
            )
        )

        rho = st.number_input(
            "Intracluster correlation rho",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            format="%.6f",
            key="power_cluster_rho",
        )

        deff = 1 + (m - 1) * rho

        st.metric("Design effect", f"{deff:.4f}")

        return float(deff), m


def power_mode(N_pop: int, use_fpc: bool, show_details: bool) -> None:
    st.header("Power-based calculation")

    if use_fpc:
        st.info(
            "Finite population correction is enabled for power calculations as an "
            "approximate unit-level adjustment after the design-effect adjustment. "
            "For complex clustered finite-population power analysis, consider a more "
            "specialized design-specific calculation."
        )
    else:
        st.info(
            "Use this mode when you want enough sample to detect a meaningful effect with "
            "a chosen level of power. This section uses standard large-sample "
            "normal-approximation formulas. Finite population correction is currently "
            "turned off."
        )

    show_power_guidance()

    st.subheader("Step 1. Choose testing assumptions")

    col1, col2, col3 = st.columns(3)

    with col1:
        alpha = st.number_input(
            "Significance level alpha",
            min_value=0.001,
            max_value=0.500,
            value=0.050,
            step=0.005,
            format="%.3f",
            help="Commonly 0.05.",
        )

    with col2:
        power = st.number_input(
            "Desired power",
            min_value=0.500,
            max_value=0.999,
            value=0.800,
            step=0.010,
            format="%.3f",
            help="Commonly 0.80 or 0.90.",
        )

    with col3:
        sided = st.selectbox(
            "Test sidedness",
            ["Two-sided", "One-sided"],
            help="Use two-sided unless there is a strong design reason to use one-sided.",
        )

    za = z_alpha(alpha, sided)
    zb = z_power(power)

    st.subheader("Step 2. Choose the outcome/test type")

    show_advanced = st.checkbox(
        "Show advanced/custom formula",
        value=False,
        help=(
            "Most users do not need this. Turn it on only if you have a custom "
            "variance factor and effect size from a separate statistical derivation."
        ),
    )

    formula_options = [
        "One-sample mean",
        "One-sample proportion",
        "Paired mean",
        "Two independent means",
        "Two independent proportions",
    ]

    if show_advanced:
        formula_options.append("Advanced: custom Wald formula")

    formula = st.selectbox("Power formula", formula_options)

    result_kind = "single"
    result_label = "Required sample size"
    n_raw = None
    n1_raw = None
    n2_raw = None

    # -----------------------------------------------------------------
    # One-sample mean
    # -----------------------------------------------------------------

    if formula == "One-sample mean":
        st.subheader("Step 3. Enter planning values")

        st.info(
            "Required inputs: expected standard deviation and the mean difference "
            "that the study should be able to detect. These can come from previous "
            "studies, past surveys, pilot data, or a meaningful policy or scientific threshold."
        )

        sigma = st.number_input(
            "Expected standard deviation",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.6f",
        )

        delta = st.number_input(
            "Mean difference to detect",
            min_value=1e-12,
            value=0.20,
            step=0.01,
            format="%.6f",
        )

        n_raw = (((za + zb) * sigma) / delta) ** 2
        result_label = "Required sample size"

    # -----------------------------------------------------------------
    # One-sample proportion
    # -----------------------------------------------------------------

    elif formula == "One-sample proportion":
        st.subheader("Step 3. Enter planning values")

        st.info(
            "Required inputs: the null proportion and the alternative proportion "
            "the study should be able to detect. These values may come from prior "
            "surveys, published studies, pilot data, or a meaningful planning target."
        )

        col1, col2 = st.columns(2)

        with col1:
            p0 = st.number_input(
                "Null proportion p0",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.6f",
            )

        with col2:
            p1 = st.number_input(
                "Alternative proportion p1",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.01,
                format="%.6f",
            )

        delta = abs(p1 - p0)

        if delta <= EPS:
            fail("p1 and p0 must be different.")

        n_raw = (
            (
                za * math.sqrt(p0 * (1 - p0))
                + zb * math.sqrt(p1 * (1 - p1))
            ) ** 2
        ) / (delta ** 2)

        result_label = "Required sample size"

    # -----------------------------------------------------------------
    # Paired mean
    # -----------------------------------------------------------------

    elif formula == "Paired mean":
        st.subheader("Step 3. Enter planning values")

        st.info(
            "Required inputs: expected standard deviation of the paired differences "
            "and the mean paired difference to detect. The SD of paired differences "
            "can come from a prior paired study, pilot study, or planning assumption."
        )

        sigma_d = st.number_input(
            "Expected SD of paired differences",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.6f",
        )

        delta = st.number_input(
            "Mean paired difference to detect",
            min_value=1e-12,
            value=0.20,
            step=0.01,
            format="%.6f",
        )

        n_raw = (((za + zb) * sigma_d) / delta) ** 2
        result_label = "Required number of pairs"

    # -----------------------------------------------------------------
    # Two independent means
    # -----------------------------------------------------------------

    elif formula == "Two independent means":
        st.subheader("Step 3. Enter planning values")

        st.info(
            "Required inputs: expected standard deviation in each group, the mean "
            "difference to detect, and the planned allocation ratio. These values can "
            "come from prior studies, historical data, pilot studies, or planning assumptions."
        )

        col1, col2 = st.columns(2)

        with col1:
            sigma1 = st.number_input(
                "Group 1 expected SD",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.6f",
            )

        with col2:
            sigma2 = st.number_input(
                "Group 2 expected SD",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.6f",
            )

        delta = st.number_input(
            "Mean difference to detect",
            min_value=1e-12,
            value=0.20,
            step=0.01,
            format="%.6f",
        )

        allocation_ratio = st.number_input(
            "Allocation ratio k = n2 / n1",
            min_value=1e-12,
            value=1.0,
            step=0.1,
            format="%.6f",
            help="Use 1.0 for equal group sizes.",
        )

        n1_raw = ((za + zb) ** 2 * (sigma1 ** 2 + sigma2 ** 2 / allocation_ratio)) / (
            delta ** 2
        )
        n2_raw = allocation_ratio * n1_raw

        result_kind = "two-arm"

    # -----------------------------------------------------------------
    # Two independent proportions
    # -----------------------------------------------------------------

    elif formula == "Two independent proportions":
        st.subheader("Step 3. Enter planning values")

        st.info(
            "Required inputs: the two proportions the study should be able to distinguish. "
            "These can come from previous surveys, prior studies, pilot data, or meaningful "
            "planning assumptions. This formula assumes equal allocation between groups."
        )

        col1, col2 = st.columns(2)

        with col1:
            p1 = st.number_input(
                "Group 1 expected proportion",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.6f",
            )

        with col2:
            p2 = st.number_input(
                "Group 2 expected proportion",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.01,
                format="%.6f",
            )

        delta = abs(p2 - p1)

        if delta <= EPS:
            fail("The two group proportions must be different.")

        p_bar = (p1 + p2) / 2

        n_each_raw = (
            (
                za * math.sqrt(2 * p_bar * (1 - p_bar))
                + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
            ) ** 2
        ) / (delta ** 2)

        n1_raw = n_each_raw
        n2_raw = n_each_raw

        result_kind = "two-arm"

    # -----------------------------------------------------------------
    # Advanced custom Wald formula
    # -----------------------------------------------------------------

    else:
        st.subheader("Step 3. Enter custom planning values")

        st.warning(
            "This advanced option is intended for users who already have a custom "
            "variance factor and effect size from a separate statistical derivation. "
            "Most users should use one of the named formulas above."
        )

        A = st.number_input(
            "Variance factor A",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.6f",
            help="The formula used is n = ((z_alpha + z_power)^2 A) / Delta^2.",
        )

        delta = st.number_input(
            "Effect size Delta",
            min_value=1e-12,
            value=0.20,
            step=0.01,
            format="%.6f",
        )

        n_raw = ((za + zb) ** 2 * A) / (delta ** 2)
        result_label = "Required sample size"

    deff, cluster_m = design_effect_widget()

    # -----------------------------------------------------------------
    # Results: single-sample or paired result
    # -----------------------------------------------------------------

    if result_kind == "single":
        assert n_raw is not None

        n_design_adjusted_raw = n_raw * deff
        fpc_population: int | None = None

        if use_fpc:
            st.subheader("Step 4. Finite population correction input")

            default_N = max(N_pop, 1)
            fpc_label = (
                "Finite number of available pairs"
                if formula == "Paired mean"
                else "Finite population size for this calculation"
            )

            fpc_population = int(
                st.number_input(
                    fpc_label,
                    min_value=1,
                    value=default_N,
                    step=1,
                    key=f"power_fpc_single_{formula}",
                )
            )

            n_adjusted_raw = fpc_adjusted_raw_n(
                n_design_adjusted_raw,
                fpc_population,
            )
            n_final = ceil_cap_to_population(n_adjusted_raw, fpc_population)

        else:
            n_adjusted_raw = n_design_adjusted_raw
            n_final = ceil_sample_size(n_adjusted_raw)

        if cluster_m is None:
            fpc_phrase = " with FPC" if use_fpc else ""

            show_final_answer(
                f"Use **{n_final:,}** as the {result_label.lower()}{fpc_phrase}. "
                "This rounded value is the main number to carry forward.",
                [(result_label, n_final)],
            )

            if not use_fpc:
                population_check_n = n_final
            else:
                population_check_n = None

        else:
            clusters = int(math.ceil(n_final / cluster_m)) if n_final > 0 else 0
            cluster_rounded_n = clusters * cluster_m

            show_final_answer(
                f"Use **{clusters:,} clusters**, giving approximately "
                f"**{cluster_rounded_n:,} sampled elements**. The design-adjusted "
                f"sample size before cluster rounding is **{n_final:,}**.",
                [
                    ("Design-adjusted sample size", n_final),
                    ("Required clusters", clusters),
                    ("Cluster-rounded elements", cluster_rounded_n),
                ],
            )

            if use_fpc and fpc_population is not None and cluster_rounded_n > fpc_population:
                st.warning(
                    f"Cluster rounding requests {cluster_rounded_n:,} elements, which "
                    f"exceeds the finite population size {fpc_population:,}. Treat this "
                    "as a census of all available units, revise the cluster size, or use "
                    "a more detailed cluster-level finite-population calculation."
                )

            population_check_n = cluster_rounded_n if not use_fpc else None

        detail_rows = [
            {"Quantity": "Alpha", "Value": alpha},
            {"Quantity": "Power", "Value": power},
            {"Quantity": "z for alpha", "Value": za},
            {"Quantity": "z for power", "Value": zb},
            {"Quantity": "Raw no-design-effect sample size", "Value": n_raw},
            {"Quantity": "Design effect", "Value": deff},
            {
                "Quantity": "Raw design-adjusted sample size before FPC",
                "Value": n_design_adjusted_raw,
            },
            {"Quantity": "FPC used", "Value": "Yes" if use_fpc else "No"},
        ]

        if use_fpc:
            detail_rows.extend(
                [
                    {"Quantity": "Finite population size for FPC", "Value": fpc_population},
                    {
                        "Quantity": "Raw design-adjusted sample size after FPC",
                        "Value": n_adjusted_raw,
                    },
                ]
            )
        else:
            detail_rows.append(
                {
                    "Quantity": "Raw design-adjusted sample size",
                    "Value": n_adjusted_raw,
                }
            )

        detail_rows.append(
            {"Quantity": "Final rounded sample size", "Value": n_final}
        )

        detail_df = pd.DataFrame(detail_rows)

        show_detail_table("Calculation details", detail_df, show_details)

        if population_check_n is not None:
            apply_population_warning(population_check_n, N_pop)

    # -----------------------------------------------------------------
    # Results: two-arm result
    # -----------------------------------------------------------------

    else:
        assert n1_raw is not None
        assert n2_raw is not None

        n1_design_adjusted_raw = n1_raw * deff
        n2_design_adjusted_raw = n2_raw * deff

        N1_fpc: int | None = None
        N2_fpc: int | None = None

        if use_fpc:
            st.subheader("Step 4. Finite population correction inputs")

            if N_pop > 1:
                default_N1 = max(1, N_pop // 2)
                default_N2 = max(1, N_pop - default_N1)
            else:
                default_N1 = 1_000
                default_N2 = 1_000

            col1, col2 = st.columns(2)

            with col1:
                N1_fpc = int(
                    st.number_input(
                        "Finite population size for group 1",
                        min_value=1,
                        value=default_N1,
                        step=1,
                        key="power_fpc_group_1",
                    )
                )

            with col2:
                N2_fpc = int(
                    st.number_input(
                        "Finite population size for group 2",
                        min_value=1,
                        value=default_N2,
                        step=1,
                        key="power_fpc_group_2",
                    )
                )

            if N_pop > 0 and (N1_fpc + N2_fpc) != N_pop:
                st.warning(
                    f"Group finite populations sum to {N1_fpc + N2_fpc:,}, but "
                    f"sidebar N = {N_pop:,}. This may be fine if the sidebar N is not "
                    "intended to be the sum of the two group populations."
                )

            n1_adjusted_raw = fpc_adjusted_raw_n(n1_design_adjusted_raw, N1_fpc)
            n2_adjusted_raw = fpc_adjusted_raw_n(n2_design_adjusted_raw, N2_fpc)

            n1_final = ceil_cap_to_population(n1_adjusted_raw, N1_fpc)
            n2_final = ceil_cap_to_population(n2_adjusted_raw, N2_fpc)

        else:
            n1_adjusted_raw = n1_design_adjusted_raw
            n2_adjusted_raw = n2_design_adjusted_raw

            n1_final = ceil_sample_size(n1_adjusted_raw)
            n2_final = ceil_sample_size(n2_adjusted_raw)

        total_final = n1_final + n2_final

        if cluster_m is None:
            fpc_phrase = " with FPC" if use_fpc else ""

            show_final_answer(
                f"Use **{n1_final:,}** in group 1 and **{n2_final:,}** in group 2"
                f"{fpc_phrase}, for a total sample size of **{total_final:,}**.",
                [
                    ("Group 1 n", n1_final),
                    ("Group 2 n", n2_final),
                    ("Total n", total_final),
                ],
            )

            population_check_n = total_final if not use_fpc else None

        else:
            clusters_1 = int(math.ceil(n1_final / cluster_m)) if n1_final > 0 else 0
            clusters_2 = int(math.ceil(n2_final / cluster_m)) if n2_final > 0 else 0

            cluster_rounded_1 = clusters_1 * cluster_m
            cluster_rounded_2 = clusters_2 * cluster_m
            cluster_rounded_total = cluster_rounded_1 + cluster_rounded_2

            show_final_answer(
                f"Use **{clusters_1:,} clusters** for group 1 and "
                f"**{clusters_2:,} clusters** for group 2. This gives approximately "
                f"**{cluster_rounded_total:,} sampled elements** in total.",
                [
                    ("Group 1 clusters", clusters_1),
                    ("Group 2 clusters", clusters_2),
                    ("Total elements", cluster_rounded_total),
                ],
            )

            if use_fpc and N1_fpc is not None and cluster_rounded_1 > N1_fpc:
                st.warning(
                    f"Group 1 cluster rounding requests {cluster_rounded_1:,} elements, "
                    f"which exceeds the group 1 finite population size {N1_fpc:,}."
                )

            if use_fpc and N2_fpc is not None and cluster_rounded_2 > N2_fpc:
                st.warning(
                    f"Group 2 cluster rounding requests {cluster_rounded_2:,} elements, "
                    f"which exceeds the group 2 finite population size {N2_fpc:,}."
                )

            cluster_df = pd.DataFrame(
                [
                    {
                        "Group": "Group 1",
                        "Final n before cluster rounding": n1_final,
                        "Clusters": clusters_1,
                        "Cluster-rounded sampled elements": cluster_rounded_1,
                    },
                    {
                        "Group": "Group 2",
                        "Final n before cluster rounding": n2_final,
                        "Clusters": clusters_2,
                        "Cluster-rounded sampled elements": cluster_rounded_2,
                    },
                    {
                        "Group": "Total",
                        "Final n before cluster rounding": total_final,
                        "Clusters": clusters_1 + clusters_2,
                        "Cluster-rounded sampled elements": cluster_rounded_total,
                    },
                ]
            )

            st.subheader("Cluster-rounded allocation")
            st.dataframe(cluster_df, hide_index=True, use_container_width=True)

            population_check_n = cluster_rounded_total if not use_fpc else None

        detail_rows = [
            {
                "Group": "Group 1",
                "Raw no-design-effect n": n1_raw,
                "Design effect": deff,
                "Raw design-adjusted n before FPC": n1_design_adjusted_raw,
                "FPC used": "Yes" if use_fpc else "No",
                "Finite population for FPC": N1_fpc if use_fpc else "Not used",
                "Raw design-adjusted n after FPC": n1_adjusted_raw,
                "Final rounded n": n1_final,
            },
            {
                "Group": "Group 2",
                "Raw no-design-effect n": n2_raw,
                "Design effect": deff,
                "Raw design-adjusted n before FPC": n2_design_adjusted_raw,
                "FPC used": "Yes" if use_fpc else "No",
                "Finite population for FPC": N2_fpc if use_fpc else "Not used",
                "Raw design-adjusted n after FPC": n2_adjusted_raw,
                "Final rounded n": n2_final,
            },
            {
                "Group": "Total",
                "Raw no-design-effect n": n1_raw + n2_raw,
                "Design effect": deff,
                "Raw design-adjusted n before FPC": (
                    n1_design_adjusted_raw + n2_design_adjusted_raw
                ),
                "FPC used": "Yes" if use_fpc else "No",
                "Finite population for FPC": (
                    (N1_fpc + N2_fpc) if use_fpc and N1_fpc is not None and N2_fpc is not None else "Not used"
                ),
                "Raw design-adjusted n after FPC": n1_adjusted_raw + n2_adjusted_raw,
                "Final rounded n": total_final,
            },
        ]

        detail_df = pd.DataFrame(detail_rows)

        show_detail_table("Calculation details", detail_df, show_details)

        if population_check_n is not None:
            apply_population_warning(
                population_check_n,
                N_pop,
                context="total sample size across groups",
            )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Sample Size Calculator",
        page_icon="📊",
        layout="wide",
    )

    st.title("Sample Size Calculator")

    st.sidebar.header("Global inputs")

    N_pop = int(
        st.sidebar.number_input(
            "Population size N",
            min_value=0,
            value=0,
            step=1,
            help=(
                "Enter 0 if population size is unknown or not relevant. "
                "If N is positive, the app can warn when a no-FPC sample size exceeds N. "
                "For SRS and stratified FPC calculations, enter a positive N."
            ),
        )
    )

    use_fpc = st.sidebar.checkbox(
        "Use finite population correction (FPC)",
        value=False,
        help=(
            "Turn this on when sampling without replacement from a finite population, "
            "especially when the sample may be a large fraction of the population."
        ),
    )

    mode = st.sidebar.radio(
        "Calculation mode",
        ["Precision / margin of error", "Power"],
    )

    show_details = st.sidebar.checkbox(
        "Open calculation details by default",
        value=False,
        help=(
            "Turn this on if you want the intermediate formulas and raw values to be "
            "expanded automatically. The final answer is always shown above the details."
        ),
    )

    with st.sidebar.expander("Notes for users", expanded=False):
        st.markdown(
            """
- The final rounded sample size shown in **Final answer** is the main figure to use.
- Intermediate raw values are mainly for transparency.
- Inputs such as SDs, proportions, ICCs, and effect sizes often come from past surveys,
  prior studies, pilot studies, administrative records, or expert judgment.
- If **FPC** is turned on:
  - SRS precision uses the sidebar population size N.
  - Stratified precision uses stratum population counts that must sum to sidebar N.
  - Equal-size cluster precision uses the number of population clusters M.
  - Power calculations use an approximate unit-level FPC after design-effect adjustment.
            """
        )

    if use_fpc and N_pop <= 0:
        st.sidebar.info(
            "FPC is enabled. SRS and stratified precision calculations require a "
            "positive sidebar N. Cluster precision and power modes may ask for additional "
            "finite-population inputs."
        )

    if mode == "Precision / margin of error":
        precision_mode(N_pop, use_fpc, show_details)
    else:
        power_mode(N_pop, use_fpc, show_details)


if __name__ == "__main__":
    main()