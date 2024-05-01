import chainlit as cl
from dotenv import load_dotenv
import textwrap

import json
from chainlit import make_async
from blendsql import blend, LLMQA, LLMJoin, LLMMap
from blendsql.models import AzureOpenaiLLM
from blendsql.db import SQLite
from research.prompts.parser_program import ParserProgram
from research.utils.database import to_serialized

load_dotenv(".env")

DB_PATH = "./research/db/hybridqa/2004_United_States_Grand_Prix_0.db"
db = SQLite(DB_PATH, check_same_thread=False)
few_shot_prompt = open("./research/prompts/hybridqa/few_shot.txt").read()
ingredients_prompt = open("./research/prompts/hybridqa/ingredients.txt").read()
serialized_db = to_serialized(db, num_rows=3)


def fewshot_parse(model, **input_program_args):
    # Dedent str args
    for k, v in input_program_args.items():
        if isinstance(v, str):
            input_program_args[k] = textwrap.dedent(v)
    res = model.predict(program=ParserProgram, **input_program_args)
    return textwrap.dedent(res["result"])


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
    parser_model = AzureOpenaiLLM("gpt-4")
    blender_model = AzureOpenaiLLM("gpt-4")

    async with cl.Step(
        name="Fewshot Parse to BlendSQL", language="sql", type="llm"
    ) as parser_step:
        parser_step.input = message.content
        blendsql_query = await make_async(fewshot_parse)(
            model=parser_model,
            ingredients_prompt=ingredients_prompt,
            few_shot_prompt=few_shot_prompt,
            serialized_db=serialized_db,
            question=message.content,
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
            silence_db_exec_errors=False,
            verbose=False,
        )
        blender_step.output = json.dumps(res.meta.prompts, indent=4)

    # Send the final answer.
    if not res.df.empty:
        await cl.Message(content="\n".join([str(i) for i in res.df.values[0]])).send()
    else:
        await cl.Message(content="Empty response.").send()
