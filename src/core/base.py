from abc import ABC, abstractmethod
import numpy as np
import time


class FrontalizationResult:
    def __init__(self, image: np.ndarray, elapsed_ms: float, info: str = ""):
        self.image = image
        self.elapsed_ms = elapsed_ms
        self.info = info


class FaceFrontalizer(ABC):

    name = "base"
    display_name = "基类"

    @abstractmethod
    def convert(self, image: np.ndarray, **kwargs) -> FrontalizationResult:
        ...

    @abstractmethod
    def get_default_params(self) -> dict:
        ...

    @abstractmethod
    def update_params(self, **kwargs) -> None:
        ...
