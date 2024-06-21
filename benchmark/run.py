from pathlib import Path
from colorama import Fore
import importlib.util
from typing import Callable
import pandas as pd


from blendsql import blend
from blendsql.models import TransformersLLM
import outlines.caching

outlines.caching.clear_cache()

MODEL = TransformersLLM("hf-internal-testing/tiny-random-PhiForCausalLM", caching=False)
NUM_ITER_PER_QUERY = 5

if __name__ == "__main__":
    print(f"Averaging based on {NUM_ITER_PER_QUERY} iterations per query...")
    print("Loading benchmarks...")
    task_to_times = {}
    for task_dir in Path(__file__).parent.iterdir():
        if not task_dir.is_dir():
            continue
        elif str(task_dir.name).startswith("__"):
            continue
        print()
        print(f"Running {task_dir.name}...")
        task_to_times[task_dir.name] = []
        spec = importlib.util.spec_from_file_location(
            "load_benchmark", str(task_dir / "load.py")
        )
        load_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_module)
        load_benchmark: Callable = load_module.load_benchmark
        db, ingredients = load_benchmark()
        for query_file in (task_dir / "queries").iterdir():
            query = open(query_file, "r").read()
            for x in range(NUM_ITER_PER_QUERY):
                print("." * x, end="\r")
                smoothie = blend(
                    query=query,
                    db=db,
                    default_model=MODEL,
                    verbose=False,
                    ingredients=ingredients,
                )
                task_to_times[task_dir.name].append(smoothie.meta.process_time_seconds)
    tasks, avg_runtime, num_queries = [], [], []
    for task_name, times in task_to_times.items():
        tasks.append(task_name)
        avg_runtime.append(sum(times) / len(times))
        num_queries.append(len(times) // NUM_ITER_PER_QUERY)
    df = pd.DataFrame(
        {"Task": tasks, "Average Runtime": avg_runtime, "# Unique Queries": num_queries}
    )
    print(
        Fore.LIGHTCYAN_EX
        + "Please paste this markdown table into your future PR"
        + Fore.RESET
    )
    print(Fore.GREEN + df.to_markdown(index=False) + Fore.RESET)
