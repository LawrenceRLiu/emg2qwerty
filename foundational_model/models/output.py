from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class AdditionalFoundationalModelOutput:
    """Additional output from the foundational model

    losses:dict[str, tuple[torch.FloatTensor, float]]: the name, loss and weight of the loss
    """
    losses:Dict[str, Tuple[float, float]]