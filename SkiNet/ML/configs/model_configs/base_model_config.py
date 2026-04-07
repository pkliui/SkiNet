from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class BaseModelConfig(BaseModel):
    """
    Base class for all model configs.

    Provides strict parsing (fails on unknown fields like YAML typos), safe mutations (validates on assignment),
    and allows fields' overrides after the config object was created.
    """
    model_config = ConfigDict(extra="forbid",
                              validate_assignment=True,
                              frozen=False)

    @property
    def required_input_multiple(self) -> IntOrTuple2d | None:
        """
        Required input height/width must be divisible by the cumulative
        downsampling factor of the encoder. For a model with stride=2
        downsampling applied `n_downsampling_layers` times, this is
        `2 ** n_downsampling_layers`.
        """
        return None
