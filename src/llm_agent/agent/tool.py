from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Type

from anthropic import BaseModel
from pydantic import create_model


class Tool(ABC):
    @property
    @abstractmethod
    def name() -> str:
        raise NotImplementedError
    
    @abstractmethod
    def schema() -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class FunctionTool(Tool):
    function: Callable[..., ...]

    @property
    def name(self) -> str:
        return self.function.__name__

    def schema(self) -> dict[str, Any]:
        function_schema = self._get_function_args_model()
        return {
            "name": self.function.__name__,
            "description": self.function.__doc__,
            "input_schema": function_schema.model_json_schema()
        } 

    def _get_function_args_model(self) -> BaseModel:
        """Imperfect but works so far"""
        param_to_type = {
            param_name: (
                param.annotation,
                param.default if param.default != inspect._empty else ...
            )
            for param_name, param in inspect.signature(self.function).parameters.items()
        }

        return create_model(
            f"{self.function.__name__}_schema",
            **param_to_type
        )


@dataclass
class FinalResultTool(Tool):
    schema_type: Type[BaseModel]

    name: str = "format_result"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "Format result",
            "input_schema": self.schema_type.model_json_schema()
        } 

    @property
    def function(self) -> Type[BaseModel]:
        def _convert(**kwargs):
            deserialized = self.schema_type.model_validate(kwargs)

            if self.schema_type.__name__ == "output":
                return deserialized.output
            
            return deserialized

        return _convert
