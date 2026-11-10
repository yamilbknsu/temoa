r"""Cap-and-trade constraints for the carbon_markets extension.

For every process (r, t) of vintage v, operational
emissions of commodity e in period p are limited per unit of capacity:

.. math::

    EM_{r,p,t,v,e} \le CAP_{r,p,t,v,e} \cdot CAPACITY_{r,p,t,v} + \textbf{CR}_{r,p,t,v,e}
    \qquad
    0 \le \textbf{CR}_{r,p,t,v,e} \le CL_{r,p,t,v,e} \cdot CAPACITY_{r,p,t,v}

where CR is the credits bought, at the fixed price ``credit_cost``, to emit above
the cap. Unused headroom under the cap earns nothing.
The cost ``credit_cost * CR`` is added, discounted, to the objective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pyomo.environ import quicksum, value

from temoa.components.costs import fixed_or_variable_cost

if TYPE_CHECKING:
    from pyomo.core.base.objective import ObjectiveData

    from temoa.extensions.carbon_markets.core.model import CarbonMarketsModel
    from temoa.types import Commodity, ExprLike, Period, Region, Technology, Vintage


def cap_and_trade_indices(
    model: CarbonMarketsModel,
) -> set[tuple[Region, Period, Technology, Vintage, Commodity]]:
    """Sparse (region, period, tech, vintage, emission) for active processes."""
    indices = set()
    for r, p, t, v, e in model.cap_and_trade_cap.sparse_keys():
        if p not in model.time_optimize:
            continue
        if (r, p, t, v) in model.capacity_var_rptv:
            indices.add((r, p, t, v, e))
    return indices


def _emission_keys(
    model: CarbonMarketsModel,
) -> dict[tuple[Region, Technology, Vintage, Commodity], list[tuple[str, str]]]:
    """Cached map (r, t, v, e) -> [(i, o)] from emission_activity."""
    cache = getattr(model, '_cap_and_trade_emission_keys', None)
    if cache is None:
        cache = {}
        for r, e, i, t, v, o in model.emission_activity.sparse_keys():
            cache.setdefault((r, t, v, e), []).append((i, o))
        model._cap_and_trade_emission_keys = cache
    return cache


def _process_emissions(
    model: CarbonMarketsModel, r: Region, p: Period, t: Technology, v: Vintage, e: Commodity
) -> ExprLike:
    expr = 0
    for i, o in _emission_keys(model).get((r, t, v, e), []):
        if (r, p, t, v) not in model.process_inputs:
            continue
        eac = value(model.emission_activity[r, e, i, t, v, o])
        if t in model.tech_annual:
            expr += eac * model.v_flow_out_annual[r, p, i, t, v, o]
        else:
            expr += eac * quicksum(
                model.v_flow_out[r, p, s, d, i, t, v, o]
                for s in model.time_season
                for d in model.time_of_day
            )
    return expr


def cap_and_trade_balance_constraint_rule(
    model: CarbonMarketsModel, r: Region, p: Period, t: Technology, v: Vintage, e: Commodity
) -> ExprLike:
    cap = value(model.cap_and_trade_cap[r, p, t, v, e])
    return (
        _process_emissions(model, r, p, t, v, e)
        <= cap * model.v_capacity[r, p, t, v] + model.v_cap_and_trade_credit[r, p, t, v, e]
    )


def cap_and_trade_credit_limit_constraint_rule(
    model: CarbonMarketsModel, r: Region, p: Period, t: Technology, v: Vintage, e: Commodity
) -> ExprLike:
    limit = value(model.cap_and_trade_credit_limit[r, p, t, v, e])
    return model.v_cap_and_trade_credit[r, p, t, v, e] <= limit * model.v_capacity[r, p, t, v]


def append_credit_costs(model: CarbonMarketsModel) -> None:
    """Add discounted credit purchases to the objective."""
    p_0 = min(model.time_optimize)
    gdr = value(model.global_discount_rate)

    credit_costs = quicksum(
        fixed_or_variable_cost(
            cap_or_flow=1.0,
            cost_factor=value(model.cap_and_trade_credit_cost[r, p, t, v, e])
            * model.v_cap_and_trade_credit[r, p, t, v, e],
            cost_years=value(model.period_length[p]),
            global_discount_rate=gdr,
            p_0=p_0,
            p=p,
        )
        for (r, p, t, v, e) in model.cap_and_trade_rptvert
    )
    model.total_cost.set_value(cast('ObjectiveData', model.total_cost).expr + credit_costs)
