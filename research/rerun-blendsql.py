from research.run_evaluate import post_process_blendsql, NpEncoder
import json
from blendsql.models import AzureOpenaiLLM
from blendsql import blend, LLMQA, LLMJoin, LLMValidate, LLMMap
from blendsql.db import SQLite
from research.constants import EvalField
from colorama import Fore
from tqdm import tqdm
import traceback
from pathlib import Path

if __name__ == "__main__":
    d = Path(
        "research/outputs/new-paper-results/hybridqa/gpt-4-blendsql-deepseek-coder"
    )
    with open(d / "predictions.json") as f:
        predictions = json.load(f)
    blender_endpoint = AzureOpenaiLLM("gpt-4")
    for idx, item in tqdm(enumerate(predictions), total=len(predictions)):
        try:
            if item["error"] is None:
                continue
            elif item["error"] == "Empty subtable passed to QAIngredient!":
                continue
            elif "maximum context length" in item["error"]:
                continue
            elif "The model attempted to generate" in item["error"]:
                continue
            db = SQLite(item[EvalField.DB_PATH])
            blendsql = item[EvalField.PRED_BLENDSQL]
            try:
                blendsql = post_process_blendsql(
                    blendsql=item[EvalField.PRED_BLENDSQL],
                    db=db,
                    use_tables=item["input_program_args"].get("use_tables", None),
                )
            except Exception as error:
                print("post process error! ")
                print(Fore.YELLOW + str(error) + Fore.RESET)
            res = blend(
                query=blendsql,
                db=db,
                ingredients={LLMMap, LLMQA, LLMJoin, LLMValidate},
                blender=blender_endpoint,
                infer_gen_constraints=True,
                silence_db_exec_errors=False,
                verbose=True,
                schema_qualify=True,
            )
            predictions[idx]["pred_has_ingredient"] = res.meta.contains_ingredient
            predictions[idx]["example_map_outputs"] = res.meta.example_map_outputs
            predictions[idx][EvalField.PREDICTION] = [
                i for i in res.df.values.flat if i is not None
            ]
            predictions[idx]["error"] = None
            print(Fore.CYAN + str(predictions[idx][EvalField.PREDICTION]) + Fore.RESET)
            print(Fore.GREEN + item[EvalField.GOLD_ANSWER] + Fore.RESET)
            print()
        except Exception as error:
            tb = traceback.format_exc()
            print(Fore.RED + tb + Fore.RESET)
            predictions[idx]["error"] = str(error)

    with open(d / "rerun-predictions.json", "w") as f:
        json.dump(predictions, f, indent=4, cls=NpEncoder)
