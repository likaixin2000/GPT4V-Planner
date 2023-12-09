from abc import ABC, abstractmethod

class PlanExecutionError(Exception):
    pass


class Environment(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def get_execution_context(self):
        pass
