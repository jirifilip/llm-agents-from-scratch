from dataclasses import dataclass, field
from typing import Generic, TypeVar

from anthropic import Anthropic, BaseModel

from llm_agent.agent.tool import Tool


AgentResultT = TypeVar("AgentResultT")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


@dataclass
class AgentResult(Generic[AgentResultT]):
    result: AgentResultT
    message_history: list[dict[str, any]]
    usage_history: list[Usage]

    @property
    def total_usage(self):
        return Usage(
            input_tokens=sum(u.input_tokens for u in self.usage_history),
            output_tokens=sum(u.output_tokens for u in self.usage_history)
        )


@dataclass
class AgentState:
    client: Anthropic
    system_prompt: str = "" 
    model_type: str = "claude-3-5-haiku-latest"
    max_tokens: int = 1000
    tools: list[Tool] = field(default_factory=list)
    message_history: list[dict[str, any]] = field(default_factory=list)
    usage_history: list["Usage"] = field(default_factory=list)