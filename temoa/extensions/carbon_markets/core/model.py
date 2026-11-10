from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pyomo.environ import BuildAction, Constraint, NonNegativeReals, Param, Set, Var

from temoa.extensions.carbon_markets.components import cap_and_trade

if TYPE_CHECKING:
    from temoa.core.model import TemoaModel

    class CarbonMarketsModel(TemoaModel):
        """TemoaModel extended with carbon-markets components for type checking."""

        cap_and_trade_cap: Param
        cap_and_trade_credit_limit: Param
        cap_and_trade_credit_cost: Param

        cap_and_trade_rptvert: Set

        v_cap_and_trade_credit: Var

        cap_and_trade_balance_constraint: Constraint
        cap_and_trade_credit_limit_constraint: Constraint

        cap_and_trade_append_cost: BuildAction


def register_model_components(model: TemoaModel) -> None:
    """Attach carbon-markets components to the core Temoa model."""
    m = cast('CarbonMarketsModel', model)

    index = (
        m.regions,
        m.time_future,
        m.tech_all,
        m.vintage_all,
        m.commodity_emissions,
    )
    m.cap_and_trade_cap = Param(*index, within=NonNegativeReals)
    m.cap_and_trade_credit_limit = Param(*index, within=NonNegativeReals)
    m.cap_and_trade_credit_cost = Param(*index, within=NonNegativeReals)

    # (region, period, tech, vintage, emission commodity)
    m.cap_and_trade_rptvert = Set(dimen=5, initialize=cap_and_trade.cap_and_trade_indices)

    # Credits bought by a process to emit above its cap
    m.v_cap_and_trade_credit = Var(m.cap_and_trade_rptvert, domain=NonNegativeReals)

    m.cap_and_trade_balance_constraint = Constraint(
        m.cap_and_trade_rptvert, rule=cap_and_trade.cap_and_trade_balance_constraint_rule
    )
    m.cap_and_trade_credit_limit_constraint = Constraint(
        m.cap_and_trade_rptvert, rule=cap_and_trade.cap_and_trade_credit_limit_constraint_rule
    )

    m.cap_and_trade_append_cost = BuildAction(rule=cap_and_trade.append_credit_costs)
