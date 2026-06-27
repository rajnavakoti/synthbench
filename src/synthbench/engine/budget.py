"""Credit budget guard for the workload engine.

Reserves estimated cost before each request is submitted and refuses once the
run would breach its threshold, so a benchmark can never silently overspend the
configured limit. Async-safe: reservations are serialized by a lock.
"""

import asyncio


class BudgetGuard:
    """Tracks cumulative reserved cost against a threshold.

    The threshold is ``limit_usd * guard_pct`` — the run halts a little before
    the hard limit so an in-flight estimate can't tip it over.
    """

    def __init__(self, limit_usd: float, guard_pct: float) -> None:
        self.limit_usd = limit_usd
        self.threshold_usd = limit_usd * guard_pct
        self._spent = 0.0
        self._lock = asyncio.Lock()
        self.exceeded = False

    @property
    def spent(self) -> float:
        return self._spent

    async def reserve(self, amount_usd: float) -> bool:
        """Reserve ``amount_usd`` if it stays within threshold.

        Returns ``True`` if reserved (caller may proceed) or ``False`` if it
        would breach the threshold (caller must skip the request). Once any
        reservation is refused, ``exceeded`` latches to ``True``.
        """
        async with self._lock:
            if self._spent + amount_usd > self.threshold_usd:
                self.exceeded = True
                return False
            self._spent += amount_usd
            return True
