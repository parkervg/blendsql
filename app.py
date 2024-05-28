from blendsql.utils import fetch_from_hub
import chainlit as cl
from chainlit import make_async
import json

from blendsql import blend, LLMQA, LLMJoin, LLMMap
from blendsql.models import OpenaiLLM
from blendsql.db import SQLite
from blendsql.prompts import FewShot
from blendsql.nl_to_blendsql import nl_to_blendsql, NLtoBlendSQLArgs

parser_model = OpenaiLLM("gpt-3.5-turbo")
blender_model = OpenaiLLM("gpt-3.5-turbo")

db = SQLite(fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db"))
few_shot_prompt = open("./research/prompts/hybridqa/few_shot.txt").read()
ingredients_prompt = open("./research/prompts/hybridqa/ingredients.txt").read()


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    async with cl.Step(
        name="Fewshot Parse to BlendSQL", language="sql", type="llm"
    ) as parser_step:
        parser_step.input = message.content
        blendsql_query = await make_async(nl_to_blendsql)(
            question=message.content,
            db=db,
            model=parser_model,
            ingredients={LLMQA, LLMMap, LLMJoin},
            few_shot_examples=FewShot.hybridqa,
            args=NLtoBlendSQLArgs(
                use_tables=["w"],
                include_db_content_tables=["w"],
                num_serialized_rows=3,
                use_bridge_encoder=True,
            ),
            verbose=False,
        )

        parser_step.output = blendsql_query

    async with cl.Step(
        name="Execute BlendSQL Script", language="json", type="llm"
    ) as blender_step:
        blender_step.input = blendsql_query
        res = await make_async(blend)(
            query=blendsql_query,
            db=db,
            ingredients={LLMMap, LLMQA, LLMJoin},
            blender=blender_model,
            infer_gen_constraints=True,
            verbose=False,
        )
        blender_step.output = json.dumps(res.meta.prompts, indent=4)

    # Send the final answer.
    if not res.df.empty:
        await cl.Message(content="\n".join([str(i) for i in res.df.values[0]])).send()
    else:
        await cl.Message(content="Empty response.").send()
