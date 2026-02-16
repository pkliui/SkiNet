from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseModelConfig(BaseModel):
    """
    Base class for all model configs.

    Provides strict parsing (fails on unknown fields like YAML typos), safe mutations (validates on assignment),
    and allows fields' overrides after the config object was created.
    """
    model_config = ConfigDict(extra="forbid",
                              validate_assignment=True,
                              frozen=False)
