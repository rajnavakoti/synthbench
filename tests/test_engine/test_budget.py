"""Tests for the credit budget guard."""

from synthbench.engine.budget import BudgetGuard


async def test_reserves_within_threshold() -> None:
    guard = BudgetGuard(limit_usd=10.0, guard_pct=1.0)
    assert await guard.reserve(4.0) is True
    assert await guard.reserve(4.0) is True
    assert guard.spent == 8.0
    assert guard.exceeded is False


async def test_refuses_over_threshold_and_latches_exceeded() -> None:
    guard = BudgetGuard(limit_usd=10.0, guard_pct=0.9)  # threshold 9.0
    assert await guard.reserve(8.0) is True
    assert await guard.reserve(2.0) is False  # 8 + 2 > 9
    assert guard.exceeded is True
    assert guard.spent == 8.0


async def test_guard_pct_lowers_threshold() -> None:
    guard = BudgetGuard(limit_usd=10.0, guard_pct=0.5)  # threshold 5.0
    assert await guard.reserve(5.0) is True
    assert await guard.reserve(0.01) is False
