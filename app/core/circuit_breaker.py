import time
from typing import Optional, List, Dict
from fastapi import HTTPException


class AsyncCircuitBreaker:
    def __init__(self, fail_max: int = 5, reset_timeout: float = 10.0, exclude: Optional[List[type]] = None):
        self.fail_max = fail_max
        self.reset_timeout = float(reset_timeout)
        self.exclude = tuple(exclude or [])
        self._fail_count = 0
        self._open_until = 0.0

    def _now(self) -> float:
        return time.time()

    def begin(self):
        if self.is_open():
            raise Exception("Circuit breaker is open")

    def is_open(self) -> bool:
        if self._open_until and self._now() < self._open_until:
            return True
        if self._open_until and self._now() >= self._open_until:
            self._open_until = 0.0
            self._fail_count = 0
        return False

    def success(self):
        self._fail_count = 0
        self._open_until = 0.0

    def failure(self, exc: Exception | None = None):
        if exc and any(isinstance(exc, ex) for ex in self.exclude):
            return
        self._fail_count += 1
        if self._fail_count >= self.fail_max:
            self._open_until = self._now() + self.reset_timeout


_model_breakers: Dict[str, AsyncCircuitBreaker] = {}


def init_breakers(model_keys: list[str]):
    global _model_breakers
    _model_breakers = {
        k: AsyncCircuitBreaker(fail_max=5, reset_timeout=10, exclude=[HTTPException]) for k in model_keys
    }


def get_breaker(model_key: str) -> AsyncCircuitBreaker:
    return _model_breakers[model_key]
