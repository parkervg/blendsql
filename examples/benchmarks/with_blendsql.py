import time
import statistics

from blendsql import blend, LLMMap
from blendsql.db import SQLite
from blendsql.llms import AzureOpenaiLLM
from blendsql.utils import fetch_from_hub
from constants import model

if __name__ == "__main__":
    """
    PYTHONPATH=$PWD:$PYTHONPATH kernprof -lv examples/benchmarks/with_blendsql.py
    """
    times = []
    iterations = 1
    db = SQLite(fetch_from_hub("multi_table.db"))
    blender = AzureOpenaiLLM(model)
    print(f"Using {model}...")
    for _ in range(iterations):
        # Uncomment if we want to clear cache first
        # guidance.llms.OpenAI.cache.clear()
        start = time.time()
        blendsql = """
        SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)'
            FROM account_history
            WHERE Symbol IN
              (
                  SELECT Symbol FROM constituents
                  WHERE sector = 'Information Technology'
                  AND {{
                          LLMMap(
                              'does this company manufacture cell phones?', 
                              'constituents::Name'
                           )
                      }} = TRUE
              )
            AND lower(Action) like "%dividend%"
        """
        smoothie = blend(
            query=blendsql,
            db=db,
            ingredients={LLMMap},
            blender=blender,
            blender_args={"few_shot": False},
            verbose=False,
        )
        runtime = time.time() - start
        print(f"Completed with_blendsql benchmark in {runtime} seconds")
        print(f"Passed {smoothie.meta.num_values_passed} total values to LLM")
        times.append(runtime)
    print(
        f"For {iterations} iterations, average runtime is {statistics.mean(times)} with stdev {statistics.stdev(times) if len(times) > 1 else 0}"
    )
