from dataclasses import dataclass, InitVar
from typing import Dict, Tuple, Optional

@dataclass
class AdditionalFoundationalModelOutput:
    """Additional output from the foundational model

    losses:dict[str, tuple[torch.FloatTensor, float]]: the name, loss and weight of the loss
    """
    losses:Optional[Dict[str, Tuple[float, float]]]