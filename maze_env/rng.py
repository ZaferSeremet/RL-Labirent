"""Her bilgisayarda ve her Python sürümünde aynı sonucu veren rastgele sayı üreteci.

Python yalnız `random.Random(seed)` tohumlamasının ve `.random()` çıktısının
sürümler arasında değişmeyeceğini garanti eder; `randint`, `shuffle`, `choice`
gibi yardımcıların algoritması sürümden sürüme değişebilir. NumPy'nin
Generator'ı da sürümler arası aynı akışı garanti etmez.

Test labirentlerinin bütün ekipte aynı olması gerektiği için labirent üretimi
yalnız `.random()` üzerine kurulu bu sınıfı kullanır.
"""

from __future__ import annotations

import random
from typing import MutableSequence, Sequence, TypeVar

T = TypeVar("T")


class StableRandom:
    def __init__(self, seed: int | None = None) -> None:
        self._r = random.Random(seed)

    def random(self) -> float:
        """[0, 1) aralığında ondalık sayı."""
        return self._r.random()

    def below(self, n: int) -> int:
        """[0, n) aralığında tam sayı."""
        if n <= 0:
            raise ValueError("n pozitif olmalı")
        return min(int(self._r.random() * n), n - 1)

    def choice(self, seq: Sequence[T]) -> T:
        return seq[self.below(len(seq))]

    def shuffle(self, seq: MutableSequence[T]) -> None:
        """Fisher-Yates, yerinde karıştırır."""
        for i in range(len(seq) - 1, 0, -1):
            j = self.below(i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def seed_int(self) -> int:
        """Yeni bir bölüm seed'i üretmek için 31 bitlik tam sayı."""
        return self.below(2**31 - 1)
