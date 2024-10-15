from abc import ABC, abstractmethod

class BaseExpert(ABC):
    @abstractmethod
    def is_mentioned(self, input: str):
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
