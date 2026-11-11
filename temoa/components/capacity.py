# temoa/components/capacity.py
"""
Defines the capacity-related components of the Temoa model.

This module is responsible for:
-  Defining Pyomo index sets for variables.
-  Defining the rules for all capacity-related constraints, such as capacity
    production limits, retirement accounting, and available capacity aggregation.
-  Pre-calculating sparse index sets for capacity, retirement, and material flows.
"""

from __future__ import annotations

from itertools import product as cross_product
from logging import getLogger
from typing import TYPE_CHECKING

from deprecated import deprecated
from pyomo.environ import value

from temoa.components import geography, technology

from .utils import get_adjusted_existing_capacity, get_available_output

if TYPE_CHECKING:
    from temoa.core.model import TemoaModel
    from temoa.types import (
        Commodity,
        ExprLike,
        Period,
        Region,
        Season,
        Technology,
        TimeOfDay,
        Vintage,
    )


logger = getLogger(name=__name__)


# ============================================================================
# HELPER FUNCTIONS AND VALIDATORS
# ============================================================================


def check_capacity_factor_process(model: TemoaModel) -> None:
    # Count of capacity factor process entries for this process in this region
    count_rtv: dict[tuple[Region, Technology, Vintage], int] = {}

    # Pull capacity_factor_tech by default
    unique_rt = {(r, t) for r, _s, _d, t in model.capacity_factor_rsdt}
    for r, t in unique_rt:
        for p in model.time_optimize:
            for v in model.process_vintages.get((r, p, t), []):
                if (r, t, v) not in count_rtv:
                    model.is_capacity_factor_process[r, t, v] = False
                    count_rtv[r, t, v] = 0

    # Check for bad values and count up the good ones
    for r, _s, _d, t, v in model.capacity_factor_process.sparse_keys():
        # Validate that vintage is active for some period
        if not model.process_periods.get((r, t, v)):
            msg = f'Invalid vintage {v} for {r, t} in capacity_factor_process table'
            logger.error(msg)
            raise ValueError(msg)

        # Good value, pull from capacity_factor_process table
        count_rtv[r, t, v] += 1

    # Check if all possible values have been set by process
    # log a warning if some are missing (allowed but maybe accidental)
    for (r, t, v), count in count_rtv.items():
        num_seg = len(model.time_season) * len(model.time_of_day)
        if count > 0:
            model.is_capacity_factor_process[r, t, v] = True
            if count < num_seg:
                logger.info(
                    'Some but not all processes were set in capacity_factor_process (%i out of a '
                    'possible %i) for: %s Missing values will default to capacity_factor_tech '
                    'value or 1 if that is not set either.',
                    count,
                    num_seg,
                    (r, t, v),
                )


@deprecated('should not be needed.  We are pulling the default on-the-fly where used')
def create_capacity_factors(model: TemoaModel) -> None:
    """
    Steps to creating capacity factors:
    1. Collect all possible processes
    2. Find the ones _not_ specified in capacity_factor_process
    3. Set them, based on capacity_factor_tech.
    """
    capacity_factor_process = model.capacity_factor_process

    # Step 1
    processes = {(r, t, v) for r, i, t, v, o in model.efficiency.sparse_keys()}

    all_cfs = {
        (r, s, d, t, v)
        for (r, t, v) in processes
        for s, d in cross_product(model.time_season, model.time_of_day)
    }

    # Step 2
    unspecified_cfs = all_cfs.difference(capacity_factor_process.sparse_keys())

    # Step 3

    # Some hackery: We futz with _constructed because Pyomo thinks that this
    # Param is already constructed.  However, in our view, it is not yet,
    # because we're specifically targeting values that have not yet been
    # constructed, that we know are valid, and that we will need.

    if unspecified_cfs:
        # CFP._constructed = False
        for r, s, d, t, v in unspecified_cfs:
            capacity_factor_process[r, s, d, t, v] = model.capacity_factor_tech[r, s, d, t]
        logger.debug(
            'Created Capacity Factors for %d processes without an explicit specification',
            len(unspecified_cfs),
        )
    # CFP._constructed = True


def get_default_capacity_factor(
    model: TemoaModel, r: Region, s: Season, d: TimeOfDay, t: Technology, v: Vintage
) -> float:
    """
    This initializer is used to fill the capacity_factor_process from the capacity_factor_tech
    where needed.

    Priority:
        1.  As specified in data input (this function not called)
        2.  Here
        3.  The default from capacity_factor_tech param
    :param M: generic model reference
    :param r: region
    :param s: season
    :param d: time-of-day slice
    :param t: tech
    :param v: vintage
    :return: the capacity factor
    """
    return value(model.capacity_factor_tech[r, s, d, t])


# ============================================================================
# PYOMO INDEX SETS
# ============================================================================


def new_capacity_variable_indices(
    model: TemoaModel,
) -> set[tuple[Region, Technology, Vintage]]:
    return model.new_capacity_rtv


def retired_capacity_variable_indices(
    model: TemoaModel,
) -> set[tuple[Region, Period, Technology, Vintage]]:
    return {
        (r, p, t, v)
        for r, p, t in model.process_vintages
        if t in model.tech_retirement and t not in model.tech_uncap
        for v in model.process_vintages[r, p, t]
        if v < p <= v + value(model.lifetime_process[r, t, v]) - value(model.period_length[p])
    }


def annual_retirement_variable_indices(
    model: TemoaModel,
) -> set[tuple[Region, Period, Technology, Vintage]]:
    return {
        (r, p, t, v)
        for r, t, v in model.retirement_periods
        for p in model.retirement_periods[r, t, v]
    }


def capacity_variable_indices(
    model: TemoaModel,
) -> set[tuple[Region, Period, Technology, Vintage]]:
    return model.active_capacity_rptv


def capacity_available_variable_indices(
    model: TemoaModel,
) -> set[tuple[Region, Period, Technology]]:
    return model.active_capacity_available_rpt


def regional_exchange_capacity_constraint_indices(
    model: TemoaModel,
) -> set[tuple[Region, Region, Period, Technology, Vintage]]:
    return {
        (r_to, r_from, p, t, v)
        for r_from, p, i in model.export_regions
        for r_to, t, v, _o in model.export_regions[r_from, p, i]
    }


def capacity_annual_constraint_indices(
    model: TemoaModel,
) -> set[tuple[Region, Period, Technology, Vintage]]:
    # Optimisation: We use the capacity annual constraint if the tech is annual
    # so we only have one capacity constraint per process per period.
    # Exceptions:
    #   - Has capacity factor tech/process constraints (which could be time-varying)
    #   - Feeds a defined DSD (so could have variable output due to DSD shape)
    cft_rt = {(r, t) for r, _s, _d, t in model.capacity_factor_tech.sparse_keys()}
    cfp_rtv = {(r, t, v) for r, _s, _d, t, v in model.capacity_factor_process.sparse_keys()}
    return {
        (r, p, t, v)
        for r, p, t, v in model.active_capacity_rptv
        if t in model.tech_annual
        and (r, t) not in cft_rt
        and (r, t, v) not in cfp_rtv
        and not any(o in model.commodity_dsd for o in model.process_outputs[r, p, t, v])
    }


def capacity_dsd_constraint_indices(
    model: TemoaModel,
) -> set[tuple[Region, Period, Season, TimeOfDay, Technology, Vintage]]:
    """Capacity rows for annual techs with DSD outputs, only in the hours that can bind."""
    return {
        (r, p, s, d, t, v)
        for (r, p, t, v), hours in annual_dsd_binding_hours(model).items()
        for s, d in hours
    }


def capacity_constraint_indices(
    model: TemoaModel,
) -> set[tuple[Region, Period, Season, TimeOfDay, Technology, Vintage]]:
    dsd_reduced = annual_dsd_binding_hours(model)
    return {
        (r, p, s, d, t, v)
        for r, p, t, v in model.active_capacity_rptv
        if (r, p, t, v) not in model.capacity_annual_constraint_rptv
        and (r, p, t, v) not in dsd_reduced
        for s in model.time_season
        for d in model.time_of_day
        if t not in model.tech_storage
    }


# ============================================================================
# BINDING-HOUR REDUCTION FOR ANNUAL TECHS WITH DSD OUTPUTS
# ============================================================================

# Relative tolerance for treating a ratio profile as flat
_FLAT_TOL = 1e-9
# Relative tolerance for the 2D hull test. A point is only dropped when it lies clearly
# inside the hull; unclear cases are kept (an extra row is redundant, never wrong).
_HULL_TOL = 1e-12


def _upper_right_hull_2d(points: list[tuple[float, float]]) -> list[int]:
    """
    Indices of the points that are needed to describe max_h (x_h*a + y_h*b) for all a, b >= 0.

    These are the vertices of the upper-right convex hull between the point with the largest x
    and the point with the largest y. Every other point is dominated by a convex combination of
    these, so its row is implied. Exact duplicates keep the first occurrence only.
    """
    # 1. Pareto filter. Sweep x descending (ties: y descending, then index ascending) and keep a
    #    point only if its y is strictly larger than everything seen so far.
    order = sorted(range(len(points)), key=lambda i: (-points[i][0], -points[i][1], i))
    pareto: list[int] = []
    best_y = float('-inf')
    for i in order:
        if points[i][1] > best_y:
            pareto.append(i)
            best_y = points[i][1]
    # now x descending, y ascending -> flip to x ascending, y descending
    pareto.reverse()
    if len(pareto) <= 2:
        return pareto

    # 2. Monotone chain over the Pareto front (left to right). Drop the middle point a of
    #    (o, a, b) when it lies strictly below the segment o -> b (clockwise turn missing).
    scale = max(max(abs(x), abs(y)) for x, y in points) or 1.0
    eps = _HULL_TOL * scale * scale
    chain: list[int] = []
    for i in pareto:
        bx, by = points[i]
        while len(chain) >= 2:
            ox, oy = points[chain[-2]]
            ax, ay = points[chain[-1]]
            cross = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
            if cross > eps:
                chain.pop()
            else:
                break
        chain.append(i)
    return chain


def annual_dsd_binding_hours(
    model: TemoaModel,
) -> dict[tuple[Region, Period, Technology, Vintage], list[tuple[Season, TimeOfDay]]]:
    r"""
    For annual techs whose outputs carry a demand-specific distribution (DSD), find the only
    time slices whose capacity constraint can bind. Computed once and cached on the model.

    An annual tech has one flow variable per (i, o) per year, so its hourly capacity row is
    (after dividing through by :math:`\text{SEG}_{s,d}`)

    .. math::
        \text{C2A}_{r,t} \cdot \textbf{CAP}_{r,p,t,v}
        \ge \sum_{I,O} \rho_{o}(s,d) \cdot \textbf{FOA}_{r,p,i,t,v,o},
        \quad \rho_{o}(s,d) = \text{DSD}_{r,p,s,d,o} / \text{SEG}_{s,d}

    (outputs that are not demands have :math:`\rho = 1`). Since the flows are non-negative,
    a row is implied whenever its :math:`\rho` vector is covered by a convex combination of
    the other rows' vectors, so only the upper-right hull of the points :math:`\rho(s,d)` is
    needed:

    - Outputs with a flat :math:`\rho` add the same term to every row and are ignored, and
      outputs with identical profiles are merged. Let k be the number of distinct profiles left.
    - k = 0: every row is the same; keep one.
    - k = 1: keep the hour with the largest :math:`\rho`.
    - k = 2: keep the vertices of the 2D upper-right hull.
    - k >= 3: not reduced; the process keeps all hourly rows in ``capacity_constraint``.

    Processes eligible are those that would use ``capacity_constraint`` only because of a DSD
    output: annual, no capacity factor entries, not curtailment or storage.

    :return: {(r, p, t, v): [(s, d), ...]} for every reduced process
    """
    if model.annual_dsd_binding_hours is not None:
        return model.annual_dsd_binding_hours

    result: dict[tuple[Region, Period, Technology, Vintage], list[tuple[Season, TimeOfDay]]] = {}
    model.annual_dsd_binding_hours = result

    hours: list[tuple[Season, TimeOfDay]] = list(
        cross_product(model.time_season, model.time_of_day)
    )
    n_hours = len(hours)
    if n_hours == 0:
        return result
    seg = [value(model.segment_fraction[s, d]) for s, d in hours]
    if min(seg) <= 0:
        # rho undefined; leave every process on the hourly path
        logger.info('Some segment_fraction values are not positive; DSD capacity reduction off.')
        return result

    cft_rt = {(r, t) for r, _s, _d, t in model.capacity_factor_tech.sparse_keys()}
    cfp_rtv = {(r, t, v) for r, _s, _d, t, v in model.capacity_factor_process.sparse_keys()}
    dsd = model.demand_specific_distribution

    # (r, p, o) -> rho profile, or None if flat
    profile_cache: dict[tuple[Region, Period, Commodity], tuple[float, ...] | None] = {}

    def profile(r: Region, p: Period, o: Commodity) -> tuple[float, ...] | None:
        key = (r, p, o)
        if key not in profile_cache:
            rho = tuple(value(dsd[r, p, s, d, o]) / seg[h] for h, (s, d) in enumerate(hours))
            hi, lo = max(rho), min(rho)
            profile_cache[key] = None if hi - lo <= _FLAT_TOL * max(abs(hi), 1.0) else rho
        return profile_cache[key]

    # set of distinct profiles -> kept hour indices (None if k >= 3)
    hull_cache: dict[frozenset[tuple[float, ...]], list[int] | None] = {}
    skipped: dict[Technology, set[Region]] = {}

    for r, p, t, v in sorted(model.active_capacity_rptv):
        if (
            t not in model.tech_annual
            or t in model.tech_curtailment
            or t in model.tech_storage
            or (r, t) in cft_rt
            or (r, t, v) in cfp_rtv
        ):
            continue
        outputs = model.process_outputs[r, p, t, v]
        if not any(o in model.commodity_dsd for o in outputs):
            continue  # handled by capacity_annual_constraint

        # non-demand outputs use SEG as coefficient, i.e. a flat rho of 1
        demand_profiles = (profile(r, p, o) for o in outputs if o in model.commodity_demand)
        profiles = frozenset(prof for prof in demand_profiles if prof is not None)
        if profiles not in hull_cache:
            k = len(profiles)
            if k == 0:
                kept: list[int] | None = [0]
            elif k == 1:
                (rho,) = profiles
                kept = [max(range(n_hours), key=rho.__getitem__)]
            elif k == 2:
                rho_a, rho_b = sorted(profiles)
                kept = sorted(_upper_right_hull_2d(list(zip(rho_a, rho_b, strict=True))))
            else:
                kept = None
            hull_cache[profiles] = kept

        kept = hull_cache[profiles]
        if kept is None:
            skipped.setdefault(t, set()).add(r)
            continue
        result[r, p, t, v] = [hours[h] for h in kept]

    for t in sorted(skipped):
        logger.warning(
            'Annual technology %s has 3 or more distinct demand-specific distribution profiles '
            'on its outputs in regions %s; using all hourly capacity rows.',
            t,
            sorted(skipped[t]),
        )
    if result:
        rows_after = sum(len(h) for h in result.values())
        logger.info(
            'Reduced hourly capacity rows for %d annual DSD processes: %d -> %d rows '
            '(%d time slices per process before, %.2f on average after).',
            len(result),
            len(result) * n_hours,
            rows_after,
            n_hours,
            rows_after / len(result),
        )
    return result


def capacity_factor_tech_indices(
    model: TemoaModel,
) -> set[tuple[Region, Season, TimeOfDay, Technology]]:
    return {
        (r, s, d, t)
        for r, _p, t in model.active_capacity_available_rpt
        for s in model.time_season
        for d in model.time_of_day
    }


@deprecated('switched over to validator... this set is typically VERY empty')
def capacity_factor_process_indices(
    model: TemoaModel,
) -> set[tuple[Region, Season, TimeOfDay, Technology, Vintage]]:
    return {
        (r, s, d, t, v)
        for r, _i, t, v, _o in model.efficiency.sparse_keys()
        for s in model.time_season
        for d in model.time_of_day
    }


# ============================================================================
# PYOMO CONSTRAINT RULES
# ============================================================================


def annual_retirement_constraint(
    model: TemoaModel, r: Region, p: Period, t: Technology, v: Vintage
) -> ExprLike:
    r"""
    Get the annualised retirement rate for a process in a given period.
    Used to output retirement (including end of life, EOL) and to model end of
    life flows and emissions. Assumes that retirement from the beginning of each period
    is evenly distributed over that model period :math:`\frac{1}{\text{LEN}_p}`
    for the accounting of retirement flows (in the same way we assume capacity is
    deployed evenly over the model period for construction inputs and embodied emissions).
    The factor :math:`\frac{\text{LSC}_{r,p,t,v}}{\text{PLF}_{r,p,t,v}}`
    adjusts the average survival during a period to the survival at the beginning
    of that period.

    .. math::
        :label: Annual Retirement

            \textbf{ART}_{r,p,t,v} =
            \begin{cases}
                \frac{1}{\text{LEN}_p} \cdot
                \frac{\text{LSC}_{r,p,t,v}}{\text{PLF}_{r,p,t,v}} \cdot \textbf{CAP}_{r,p,t,v}
                & \text{if EOL} \\
                \frac{1}{\text{LEN}_p} \cdot
                \left(
                \frac{\text{LSC}_{r,p,t,v}}{\text{PLF}_{r,p,t,v}} \cdot \textbf{CAP}_{r,p,t,v}
                - \frac{\text{LSC}_{r,p_{next},t,v}}{\text{PLF}_{r,p_{next},t,v}} \cdot
                \textbf{CAP}_{r,p_{next},t,v}
                \right)
                & \text{otherwise} \\
            \end{cases}

            \\\text{where EOL when } p \leq v + LTP_{r,t,v} < p + LEN_p
    """

    p_end = p + value(model.period_length[p])
    eol_year = v + value(model.lifetime_process[r, t, v])

    ## Get the capacity at the start of this period
    if p == eol_year:
        # Exact EOL. No v_capacity or v_retired_capacity for this period.
        if p == model.time_optimize.first():
            # Must be existing capacity. Apply survival curve to existing cap
            cap_begin = get_adjusted_existing_capacity(model, r, t, v) * value(
                model.lifetime_survival_curve[r, p, t, v]
            )
        else:
            # Get previous capacity and continue survival curve
            p_prev = model.time_optimize.prev(p)
            cap_begin = (
                model.v_capacity[r, p_prev, t, v]
                * value(model.lifetime_survival_curve[r, p, t, v])
                / value(model.process_life_frac[r, p_prev, t, v])
            )
    else:
        # The capacity at the beginning of the period
        cap_begin = (
            model.v_capacity[r, p, t, v]
            * value(model.lifetime_survival_curve[r, p, t, v])
            / value(model.process_life_frac[r, p, t, v])
        )

    ## Get the capacity at the end of this period
    if p <= eol_year < p_end:
        # EOL so capacity ends on zero no matter what
        cap_end = 0
    elif p == model.time_optimize.last() or p_end == eol_year:
        # No v_capacity or v_retired_capacity for next period so just continue down the
        # survival curve. If eol_year = p_end then eol would be dumped in the next period
        cap_end = (
            cap_begin
            * value(model.lifetime_survival_curve[r, p_end, t, v])
            / value(model.lifetime_survival_curve[r, p, t, v])
        )
    else:
        # Get the next period's beginning capacity
        p_next = model.time_optimize.next(p)
        cap_end = (
            model.v_capacity[r, p_next, t, v]
            * value(model.lifetime_survival_curve[r, p_next, t, v])
            / value(model.process_life_frac[r, p_next, t, v])
        )
        # next v_capacity also accounts for decision retirement so need to undo that again
        p_next_end = p_next + value(model.period_length[p_next])
        if t in model.tech_retirement and v < p_next and p_next_end <= eol_year:
            cap_end += model.v_retired_capacity[r, p_next, t, v]

    # v_capacity already accounts for decision retirement so need to undo that for beginning cap
    if t in model.tech_retirement and v < p and p_end <= eol_year:
        cap_begin += model.v_retired_capacity[r, p, t, v]

    annualised_retirement = (cap_begin - cap_end) / value(model.period_length[p])
    return model.v_annual_retirement[r, p, t, v] == annualised_retirement


def capacity_available_by_period_and_tech_constraint(
    model: TemoaModel, r: Region, p: Period, t: Technology
) -> ExprLike:
    r"""

    The :math:`\textbf{CAPAVL}` variable is nominally for reporting solution values,
    but is also used in the Limit constraint calculations.

    .. math::
        :label: CapacityAvailable

        \textbf{CAPAVL}_{r, p, t} = \sum_{v, p_i \leq p} \textbf{CAP}_{r, p, t, v}

        \\
        \forall p \in \text{P}^o, r \in R, t \in T
    """
    cap_avail = sum(model.v_capacity[r, p, t, S_v] for S_v in model.process_vintages[r, p, t])

    expr = model.v_capacity_available_by_period_and_tech[r, p, t] == cap_avail
    return expr


def capacity_annual_constraint(
    model: TemoaModel, r: Region, p: Period, t: Technology, v: Vintage
) -> ExprLike:
    r"""
    Similar to Capacity_constraint, but for technologies belonging to the
    :code:`tech_annual`  set. Technologies in the tech_annual set have constant output
    across different timeslices within a year, so we do not need to ensure
    that installed capacity is sufficient across all timeslices, thus saving
    some computational effort. Instead, annual output is sufficient to calculate
    capacity. Hourly capacity factors cannot be defined to annual technologies
    but annual capacity factors can be set using limit_annual_capacity_factor,
    which will be implicitly accounted for here.

    .. math::
        :label: CapacityAnnual

            \text{C2A}_{r, t}
            \cdot \textbf{CAP}_{r, p, t, v}
        \ge
            \sum_{I, O} \textbf{FOA}_{r, p, i, t \in T^{a}, v, o}

        \\
        \forall \{r, p, t \in T^{a}, v\} \in \Theta_{\text{Activity}}
    """
    activity_rptv = sum(
        model.v_flow_out_annual[r, p, S_i, t, v, S_o]
        for S_i in model.process_inputs[r, p, t, v]
        for S_o in model.process_outputs_by_input[r, p, t, v, S_i]
    )

    return value(model.capacity_to_activity[r, t]) * model.v_capacity[r, p, t, v] >= activity_rptv


def capacity_constraint(
    model: TemoaModel, r: Region, p: Period, s: Season, d: TimeOfDay, t: Technology, v: Vintage
) -> ExprLike:
    r"""
    This constraint ensures that the capacity of a given process is sufficient
    to support its activity across all time periods and time slices. The calculation
    on the left hand side of the equality is the maximum amount of energy a process
    can produce in the timeslice :code:`(s,d)`. Note that the curtailment variable
    shown below only applies to technologies that are members of the curtailment set.
    Curtailment is necessary to track explicitly in scenarios that include a high
    renewable target. Without it, the model can generate more activity than is used
    to meet demand, and have all activity (including the portion curtailed) count
    towards the target. Tracking activity and curtailment separately prevents this
    possibility.

    .. math::
       :label: Capacity

           \left (
                   \text{CFP}_{r, s, d, t, v}
             \cdot \text{C2A}_{r, t}
             \cdot \text{SEG}_{s, d}
           \right )
           \cdot \textbf{CAP}_{r, p, t, v}
           =
           \sum_{I, O} \textbf{FO}_{r, p, s, d, i, t, v, o}
           +
           \sum_{I, O} \textbf{CUR}_{r, p, s, d, i, t, v, o}

       \\
       \forall \{r, p, s, d, t, v\} \in \Theta_{\text{FO}}
    """
    # The expressions below are defined in-line to minimize the amount of
    # expression cloning taking place with Pyomo.

    if t in model.tech_annual:
        # Annual demand technology
        useful_activity = sum(
            (
                value(model.demand_specific_distribution[r, p, s, d, S_o])
                if S_o in model.commodity_demand
                else value(model.segment_fraction[s, d])
            )
            * model.v_flow_out_annual[r, p, S_i, t, v, S_o]
            for S_i in model.process_inputs[r, p, t, v]
            for S_o in model.process_outputs_by_input[r, p, t, v, S_i]
        )
    else:
        useful_activity = sum(
            model.v_flow_out[r, p, s, d, S_i, t, v, S_o]
            for S_i in model.process_inputs[r, p, t, v]
            for S_o in model.process_outputs_by_input[r, p, t, v, S_i]
        )

    if t in model.tech_curtailment:
        # If technologies are present in the curtailment set, then enough
        # capacity must be available to cover both activity and curtailment.
        curtailment = sum(
            model.v_curtailment[r, p, s, d, S_i, t, v, S_o]
            for S_i in model.process_inputs[r, p, t, v]
            for S_o in model.process_outputs_by_input[r, p, t, v, S_i]
        )
        return get_available_output(model, r, p, s, d, t, v) == useful_activity + curtailment
    else:
        return get_available_output(model, r, p, s, d, t, v) >= useful_activity


def adjusted_capacity_constraint(
    model: TemoaModel, r: Region, p: Period, t: Technology, v: Vintage
) -> ExprLike:
    r"""
    This constraint updates the capacity of a process by taking into account retirements
    and end of life. For a given :code:`(r,p,t,v)` index, this constraint sets the capacity
    equal to the amount installed in period :code:`v` and subtracts from it any and all retirements
    that occurred prior to the period in question, :code:`p`, and end of life from the
    survival curve if defined. It finally adjusts for the process life fraction, which
    accounts for a possible mid-period end of life where, for example, EOL 3 years into a 5-year
    period would be treated as :math:`\frac{3}{5}` capacity for all 5 years.

    .. figure:: images/adjusted_capacity_plf.*
        :align: center
        :width: 100%
        :figclass: align-center
        :figwidth: 50%

        For processes reaching end of life mid-period, the process life fraction adjustment is
        applied, distributing the effective capacity over the whole period.

    For processes using survival curves, the yearly survival curve :math:`\text{LSC}_{r,p,t,v}` is
    averaged over the period to get the effective remaining capacity for that period  Because this
    implicitly handles mid-period end of life, :math:`\text{PLF}_{r,p,t,v}` is used to account for
    both phenomena.

    .. figure:: images/adjusted_capacity_sc.*
        :align: center
        :width: 100%
        :figclass: align-center
        :figwidth: 50%

        For processes with a defined survival curve, the surviving capacity is averaged over each
        period to get the adjusted capacity. This implicitly handles mid-period end of life as a
        survival curve will always be zero after the end of life of a process.

    .. math::
        :label: Adjusted Capacity

            \textbf{CAP}_{r,p,t,v} =
            \begin{cases}
                \text{PLF}_{r,p,t,v} \cdot
                \left(
                \text{ECAP}_{r,t,v} - \sum\limits_{v < p' <= p}
                \frac{\textbf{RCAP}_{r,p',t,v}}{\text{LSC}_{r,p',t,v}}
                \right)
                & \text{if } \ v \in T^e \\
                \text{PLF}_{r,p,t,v} \cdot
                \left(
                \textbf{NCAP}_{r,t,v} - \sum\limits_{v < p' <= p}
                \frac{\textbf{RCAP}_{r,p',t,v}}{\text{LSC}_{r,p',t,v}}
                \right)
                & \text{if } \ v \notin T^e
            \end{cases}

            \\\text{where }
            \text{PLF}_{r,p,t,v} =
            \begin{cases}
                \frac{1}{\text{LEN}_p} \cdot \left(
                \sum\limits_{y = p}^{p+\text{LEN}_{p}-1}{\text{LSC}_{r,y,t,v}}
                \right)
                & \text{if } t \in T^{sc} \\
                \frac{1}{\text{LEN}_p} \cdot \left( v + \text{LTP}_{r,t,v} - p \right)
                & \text{if } t \notin T^{sc} \\
            \end{cases}

    We divide :math:`\frac{\textbf{RCAP}_{r,p',t,v}}{\text{LSC}_{r,p',t,v}}`
    because the average survival factor in :math:`\text{PLF}_{r,p,t,v}` is indexed to the vintage
    period (the beginning of the survival curve). So, we adjust for the relative survival from
    the time when that retirement occurred (treated here as at the beginning of each period).
    """

    built_capacity: ExprLike
    if v in model.time_exist:
        built_capacity = get_adjusted_existing_capacity(model, r, t, v)
    else:
        built_capacity = model.v_new_capacity[r, t, v]

    early_retirements = 0
    if t in model.tech_retirement:
        early_retirements = sum(
            model.v_retired_capacity[r, S_p, t, v]
            / value(model.lifetime_survival_curve[r, S_p, t, v])  # relative survival since then
            for S_p in model.time_optimize
            if v < S_p <= p
            and S_p + value(model.period_length[S_p]) <= v + value(model.lifetime_process[r, t, v])
        )

    remaining_capacity = (built_capacity - early_retirements) * value(
        model.process_life_frac[r, p, t, v]
    )
    return model.v_capacity[r, p, t, v] == remaining_capacity


# ============================================================================
# PRE-COMPUTATION FUNCTIONS
# ============================================================================


def create_capacity_and_retirement_sets(model: TemoaModel) -> None:
    """
    Creates and populates component-specific Python sets and dictionaries on the model object.

    This function is called once during model initialization and is responsible for
    creating the sparse indices related to technology capacity, retirement, and
    construction/end-of-life material flows. These data structures are then
    used by other functions in this module to build Pyomo components.

    Populates:
        - model.retirement_periods: dict mapping (r, t, v) to a set of periods `p`
          where retirement can occur.
        - model.capacity_consumption_techs: dict mapping (r, v, i) to a set of techs `t`
          that consume commodity `i` for construction.
        - model.retirement_production_processes: dict mapping (r, p, o) to a set of `(t, v)`
          processes that produce commodity `o` at end-of-life.
        - model.new_capacity_rtv: set of (r, t, v) for new capacity investments.
        - model.active_capacity_available_rpt: set of (r, p, t) where capacity is active.
        - model.active_capacity_available_rptv: set of (r, p, t, v) where vintage capacity is
          active.
    """

    logger.debug('Creating capacity, retirement, and construction/EOL sets.')
    # Calculate retirement periods based on lifetime and survival curves
    unique_rtv = {(r, t, v) for r, _i, t, v, _o in model.efficiency.sparse_keys()} | set(
        model.existing_capacity.sparse_keys()
    )
    for r, t, v in unique_rtv:
        if t in model.tech_uncap:
            # No capacity to retire
            continue
        if t not in model.tech_all:
            # Not an active technology so wont have a lifetime
            # If it has an EOLoutput it will be in tech_all
            continue
        lifetime = value(model.lifetime_process[r, t, v])
        for p in model.time_optimize:
            # retires bang on start of horizon or survives into planning periods
            is_p0_eol = (
                (p == model.time_optimize.first())
                and (v + lifetime == p)
                and value(model.existing_capacity[r, t, v]) > 0
            )
            is_living = (r, t, v) in model.process_periods
            if not (is_p0_eol or is_living):
                continue

            is_natural_eol = p <= v + lifetime < p + value(model.period_length[p])
            is_early_retire = t in model.tech_retirement and v < p <= v + lifetime - value(
                model.period_length[p]
            )
            is_survival_curve = model.is_survival_curve_process[r, t, v] and v <= p <= v + lifetime

            if any((is_natural_eol, is_early_retire, is_survival_curve)):
                model.retirement_periods.setdefault((r, t, v), set()).add(p)

    # Link construction materials to technologies
    for r, i, t, v in model.construction_input.sparse_keys():
        model.capacity_consumption_techs.setdefault((r, v, i), set()).add(t)
        model.used_techs.add(t)

    # Link end-of-life materials to retiring technologies
    for r, t, v, o in model.end_of_life_output.sparse_keys():
        if (r, t, v) in model.retirement_periods:
            for p in model.retirement_periods[r, t, v]:
                model.retirement_production_processes.setdefault((r, p, o), set()).add((t, v))
                model.used_techs.add(t)

    # Link end-of-life emissions to retiring technologies
    for r, _e, t, v in model.emission_end_of_life.sparse_keys():
        if (r, t, v) in model.retirement_periods:
            model.used_techs.add(t)

    # Create active capacity index sets from the now-populated process_vintages
    model.new_capacity_rtv = {
        (r, t, v)
        for r, t, v in model.process_periods
        if t not in model.tech_uncap and v in model.time_optimize
    }
    model.active_capacity_available_rpt = {
        (r, p, t) for r, p, t in model.process_vintages if t not in model.tech_uncap
    }
    model.active_capacity_rptv = {
        (r, p, t, v)
        for r, p, t in model.active_capacity_available_rpt
        for v in model.process_vintages[r, p, t]
    }


def gather_group_active_processes(
    model: TemoaModel, r: Region, p: Period, t: Technology
) -> set[tuple[Region, Technology]]:
    """
    A utility to get the valid active processes for a group-region, tech-group pair in period p.
    Caches the result if its a new call so we don't repeat these O(n2) lookups.
    """
    if (r, p, t) not in model.group_active_processes:
        model.group_active_processes[r, p, t] = {
            (_r, _t)
            for _r in geography.gather_group_regions(model, r)
            for _t in technology.gather_group_techs(model, t)
            if (_r, p, _t) in model.process_vintages
        }
    return model.group_active_processes[r, p, t]


def gather_group_built_processes(
    model: TemoaModel, r: Region, t: Technology, v: Vintage
) -> set[tuple[Region, Technology]]:
    """
    A utility to get the valid built processes for a group-region, tech-group, vintage index.
    Caches the result if its a new call so we don't repeat these O(n2) lookups.
    """
    if (r, t, v) not in model.group_built_processes:
        model.group_built_processes[r, t, v] = {
            (_r, _t)
            for _r in geography.gather_group_regions(model, r)
            for _t in technology.gather_group_techs(model, t)
            if (_r, _t, v) in model.process_periods
        }
    return model.group_built_processes[r, t, v]
