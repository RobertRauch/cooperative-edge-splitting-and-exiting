from typing import NamedTuple


class TaskSize(NamedTuple):
    request_size: float
    response_size: float
    instruction_count: float
    instruction_count_ue: float = 0
