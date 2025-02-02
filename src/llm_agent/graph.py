from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from dataclasses import dataclass


StateType = TypeVar("StateType")
ReturnType = TypeVar("ResultType")


@dataclass
class Context(Generic[StateType]):
    state: StateType


@dataclass
class End(Generic[ReturnType]):
    result: ReturnType


class Node(ABC, Generic[StateType, ReturnType]):
    @abstractmethod
    def run(self, ctx: Context[StateType]) -> "Node[StateType, ReturnType]" | End[ReturnType]:
        raise NotImplementedError
    

@dataclass
class Graph(Generic[StateType, ReturnType]):
    nodes: list[Node[StateType, ReturnType]]
    max_steps: int = 5

    def run(self, start: Node[StateType, ReturnType], state: StateType = None) -> tuple[ReturnType, list[Node[StateType, ReturnType] | End[ReturnType]]]:
        ctx = Context(state)
        
        next_node = start
        steps = []

        while True:
            steps.append(next_node)
            next_node = next_node.run(ctx)

            if len(steps) >= self.max_steps:
                raise ValueError("Max steps reached")

            if isinstance(next_node, End):
                return next_node.result, steps
            
            if not isinstance(next_node, tuple(self.nodes)):
                raise ValueError(f"Uknown step: {type(next_node)}. Please define in graph nodes first.")
