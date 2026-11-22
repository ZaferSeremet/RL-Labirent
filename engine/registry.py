"""agents/ ve baselines/ klasörlerindeki ajan sınıflarını otomatik bulur."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field

from agents.base import Agent

PACKAGES = ("agents", "baselines")


@dataclass
class Registry:
    agents: dict[str, type[Agent]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)  # modül adı -> hata

    def get(self, name: str) -> type[Agent]:
        if name not in self.agents:
            raise KeyError(f"'{name}' adlı ajan yok. Var olanlar: {', '.join(sorted(self.agents))}")
        return self.agents[name]

    def baselines(self) -> list[type[Agent]]:
        return [c for c in self.agents.values() if c.is_baseline]

    def learners(self) -> list[type[Agent]]:
        return [c for c in self.agents.values() if not c.is_baseline]


def discover() -> Registry:
    """Bir modül hatalıysa diğerleri yine yüklenir; hata `errors` içinde döner."""
    reg = Registry()
    for pkg_name in PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for mod_info in pkgutil.iter_modules(pkg.__path__):
            if mod_info.name in ("base",) or mod_info.name.startswith("_"):
                continue
            mod_name = f"{pkg_name}.{mod_info.name}"
            try:
                module = importlib.import_module(mod_name)
            except Exception as e:  # noqa: BLE001 - arkadaşın dosyasındaki hata diğerlerini durdurmasın
                reg.errors[mod_name] = f"{type(e).__name__}: {e}"
                continue
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, Agent) and cls is not Agent and cls.__module__ == mod_name:
                    if cls.name in reg.agents and reg.agents[cls.name] is not cls:
                        reg.errors[mod_name] = (
                            f"'{cls.name}' adı {reg.agents[cls.name].__module__} içinde de var"
                        )
                        continue
                    reg.agents[cls.name] = cls
    return reg
