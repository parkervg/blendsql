from blendsql.db import SQLite
from research.constants import EvalField
from blendsql.models import AzureOpenaiLLM
from tqdm import tqdm
import json
from blendsql import LLMQA, LLMMap, LLMJoin, blend
from blendsql._program import Program
import time
from research.run_evaluate import NpEncoder

BASE_SYSTEM_PROMPT = """
This is a hybrid question answering task. The goal of this task is to answer the question given a table (`w`) and corresponding passages (`docs`).
Be as succinct as possible in answering the given question, do not include explanation.
"""

with open("./research/prompts/hybridqa/few_shot.txt", "r") as f:
    few_shot_prompt = f.read()

with open("./research/prompts/hybridqa/ingredients.txt", "r") as f:
    ingredients_prompt = f.read()


class ParserProgram(Program):
    def __call__(
        self, serialized_db: str, question: str, bridge_hints: str = None, **kwargs
    ):
        with self.systemcontext:
            self.model += BASE_SYSTEM_PROMPT.format(
                ingredients_prompt=ingredients_prompt
            )
        with self.usercontext:
            self.model += f"{few_shot_prompt}\n\n"
            self.model += f"{serialized_db}\n\n"
            if bridge_hints:
                self.model += (
                    f"Here are some values that may be useful: {bridge_hints}\n"
                )
            self.model += f"Q: {question}\n"
            self.model += f"BlendSQL:\n"
        return self.model._current_prompt()


with open(
    "./research/outputs/paper-results/hybridqa/gpt-4-blendsql-plus-pp/predictions.json",
    "r",
) as f:
    predictions = json.load(f)

if __name__ == "__main__":
    parser_model = AzureOpenaiLLM("gpt-4")
    blender_model = AzureOpenaiLLM("gpt-4", caching=False)
    start = time.time()
    results = []
    for idx, item in enumerate(tqdm(predictions)):
        num_tokens = 0
        try:
            blender_model.num_prompt_tokens = 0
            res = None
            db = SQLite(item[EvalField.DB_PATH])
            question = item[EvalField.QUESTION]
            prompt = ParserProgram(
                model=parser_model.model,
                serialized_db=db.to_serialized(num_rows=3),
                question=question,
            )
            num_tokens += len(parser_model.tokenizer.encode(prompt))
            try:
                res = blend(
                    query=item["pred_sql"],
                    db=db,
                    ingredients={LLMMap, LLMQA, LLMJoin},
                    blender=blender_model,
                    infer_gen_constraints=True,
                    silence_db_exec_errors=False,
                    verbose=False,
                )
                num_tokens += res.meta.num_prompt_tokens
            except:
                pass
        except:
            pass
        finally:
            answer = [""]
            if res is not None:
                answer = [i for i in res.df.values.flat if i is not None]
            predictions[idx][EvalField.PREDICTION] = answer
            predictions[idx]["num_tokens"] = num_tokens
            if idx % 100 == 0:
                with open("num-tokens.json", "w") as f:
                    json.dump(predictions, f, indent=4, cls=NpEncoder)
    with open("num-tokens.json", "w") as f:
        json.dump(predictions, f, indent=4, cls=NpEncoder)

    #
    # print("Average: BlendSQL, 3 rows:")
    # print(num_tokens)
    # print(num_tokens / len(predictions))
    # with open('answers.json', 'w') as f:
    #     json.dump(f, results, indent=4)
    # with open('num-tokens.json', 'w') as f:
    #     json.dump(f, {"num_tokens": num_tokens, "average": num_tokens / len(predictions)}, indent=4)
