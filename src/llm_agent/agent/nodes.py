from dataclasses import dataclass
from typing import Any

from llm_agent.agent.types import AgentState, Usage
from llm_agent.agent.utils import select_from_dict, wrap_into_list
from llm_agent.graph import Context, End, Node
from llm_agent.agent.tool import FinalResultTool


@dataclass
class UseTool(Node[AgentState, dict[str, Any]]):
    tool_name: str
    tool_use_id: str
    tool_args: dict[str, Any]

    def run(self, ctx: Context[AgentState]) -> "SendMessage":
        tool = next(filter(lambda t: t.name == self.tool_name, ctx.state.tools))
        
        tool_result = tool.function(**self.tool_args)

        if self.tool_name == FinalResultTool.name:
            ctx.state.message_history.pop()
            return End(tool_result)

        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.tool_use_id,
                    "content": tool_result
                }
            ]
        }

        return SendMessage(message=message)

@dataclass
class SendMessage(Node[AgentState, dict[str, any]]):
    message: dict[str, any] | list[dict[str, any]]
    
    def run(self, ctx: Context[AgentState]) -> UseTool | End:
        prepared_messages = wrap_into_list(self.message)

        response_message = ctx.state.client.messages.create(
            max_tokens=ctx.state.max_tokens,
            messages=ctx.state.message_history + prepared_messages,
            tools=[
                tool.schema() for tool in ctx.state.tools
            ],
            system=ctx.state.system_prompt,
            model=ctx.state.model_type
        )

        ctx.state.message_history.extend(prepared_messages)
        ctx.state.message_history.append(
            select_from_dict(
                response_message.model_dump(),
                ["role", "content"]
            )
        )
        ctx.state.usage_history.append(
            Usage.model_validate(response_message.usage.model_dump())
        )

        if response_message.stop_reason == "tool_use":
            tool_block = response_message.content[-1]
            return UseTool(tool_block.name, tool_block.id, tool_block.input)

        return End(response_message)