from abc import ABC, abstractmethod
import random


class ImageIdGenerator(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass


class RandomIdGenerator(ImageIdGenerator):
    def __init__(self, max_id=9999):
        self.max_id = max_id

    def get_id(self) -> int:
        return random.randint(0, self.max_id)

