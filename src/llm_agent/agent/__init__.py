from dataclasses import dataclass, field
import inspect
from PIL import Image
from typing import Any, Type

from anthropic import Anthropic
from pydantic import BaseModel, create_model

from llm_agent.agent.nodes import SendMessage, UseTool
from llm_agent.agent.types import AgentResult, AgentState
from llm_agent.agent.utils import convert_image_to_base64_string
from llm_agent.graph import Graph
from llm_agent.agent.tool import FinalResultTool, Tool


@dataclass
class Agent:
    client: Anthropic
    result_type: Type[BaseModel]
    tools: list[Tool] = field(default_factory=list)
    system_prompt: str = ""
    model_type: str = "claude-3-5-haiku-latest"

    _GRAPH = Graph([SendMessage, UseTool], max_steps=5)
    
    def run(self, prompt: str, message_history: list[dict[str, Any]] | None = None) -> AgentResult:
        input_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }

        return self._run([input_message], message_history=message_history)
    
    def run_with_images(
            self,
            prompt: str,
            images: list[Image.Image],
            message_history: list[dict[str, Any]] | None = None
        ) -> AgentResult:
        image_format = "jpeg"
        image_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{image_format}",
                    "data": convert_image_to_base64_string(image, image_format)
                }
            } for image in images
        ]

        input_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_content
            ]
        }

        return self._run([input_message], message_history=message_history)

    def _run(
            self,
            messages: list[dict[str, any]],
            message_history: list[dict[str, Any]] | None = None
        ) -> AgentResult:
        prepared_result_type = self._preprocess_result_type(self.result_type)
        final_result_tool = FinalResultTool(prepared_result_type)
        
        state = AgentState(
            self.client,
            tools=self.tools + [final_result_tool],
            system_prompt=inspect.cleandoc(self.system_prompt),
            message_history=message_history if message_history else [],
            model_type=self.model_type
        )

        start_node = SendMessage(messages)
        graph_result = self._GRAPH.run(start_node, state)
        llm_result = graph_result[0]

        return AgentResult(
            result=llm_result,
            message_history=state.message_history,
            usage_history=state.usage_history
        )

    @staticmethod
    def _preprocess_result_type(result_type):
        if isinstance(result_type, type) and issubclass(result_type, BaseModel):
            return result_type
        
        return create_model(
            "output",
            output=(result_type, ...)
        )