# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% vscode={"languageId": "plaintext"}
import datetime
from anthropic import Anthropic, BaseModel
from pydantic import Field

from llm_agent.agent import Agent
from llm_agent.agent.tool import FinalResultTool, FunctionTool

from PIL import Image

# %%
client = Anthropic()


# %% [markdown]
# Obviously the system prompt passing is not great but that can always be abstracted away.

# %% [markdown]
# ## Greeting Agent

# %%
class GreetingResult(BaseModel):
    person_name: str | None = None
    person_occupation: str | None = None
    proper_greeting: str | None = Field(None, description="Proper greeting for a person")


def greet_person(name: str, occupation: str) -> str:
    """Greet a person"""
    
    return f"Nazdar, {name}"


# %%
greeting_agent = Agent(
    client,
    result_type=GreetingResult,
    tools=[FunctionTool(greet_person)],
    model_type="claude-3-5-haiku-latest",
    system_prompt=(
    f"""
    You are an expert entity extractor.

    <instructions>
    - use {FinalResultTool.name} to check your result.
    - do not use introduce tool usage with additional messages
    </instructions>
    """)
)

# %%
greeting_result = greeting_agent.run("My name is Jirka and I work as a programmer.")

# %%
greeting_result.result
# >> GreetingResult(person_name='Jirka', person_occupation='programmer', proper_greeting='Nazdar, Jirka')

# %% [markdown]
# ## Primitives extraction

# %%
hobby_agent = Agent(
    client,
    result_type=str,
    model_type="claude-3-5-haiku-latest",
    system_prompt=(
    f"""
    You are an expert entity extractor.

    Extract favorite hobby of the user.

    Use {FinalResultTool.name} to check your result.
    """)
);

# %%
hobby_agent.run("Hello, I like to sew").result
# >> 'sewing'

# %%
age_agent = Agent(
    client,
    result_type=int,
    model_type="claude-3-5-haiku-latest",
    system_prompt=(
    f"""
    You are an expert entity extractor.

    Extract age of the user.

    Use {FinalResultTool.name} to check your result.
    """)
);

# %%
age_agent.run("In two year I'll be twenty").result
# >> 18

# %% [markdown]
# ## Extraction from images

# %%
class Clothing(BaseModel):
    type: str
    color: str
    neck_type: str


# %%
clothing_agent = Agent(
    client,
    result_type=list[Clothing],
    model_type="claude-3-5-sonnet-latest",
    system_prompt=(
    f"""
    You are an expert entity extractor.

    Use {FinalResultTool.name} to check your result.
    """)
);

# %%
images = [
    Image.open("../data/long_sleeve.jpg"),
    Image.open("../data/short_sleeve.jpg"),
]

# %%
clothing_agent.run_with_images("Extract what is on the image", images).result
# >> [Clothing(type='sweater', color='red', neck_type='turtleneck'),
#     Clothing(type='t-shirt', color='white', neck_type='crew neck')]
