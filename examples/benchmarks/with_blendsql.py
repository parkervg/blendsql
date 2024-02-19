import time
import statistics
import guidance

from blendsql import blend, SQLiteDBConnector, init_secrets
from blendsql.ingredients import LLM

init_secrets("secrets.json")

if __name__ == "__main__":
    """
    PYTHONPATH=$PWD:$PYTHONPATH kernprof -lv examples/benchmarks/with_blendsql.py
    """
    times = []
    iterations = 10
    db_path = "./tests/multi_table.db"
    db = SQLiteDBConnector(db_path=db_path)
    for _ in range(iterations):
        # Uncomment if we want to clear cache first
        guidance.llms.OpenAI.cache.clear()
        start = time.time()
        blendsql = """
        SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)'
            FROM account_history
            WHERE Symbol IN
              (
                  SELECT Symbol FROM constituents
                  WHERE sector = 'Information Technology'
                  AND {{
                          LLM(
                              'does this company manufacture cell phones?', 
                              'constituents::Name', 
                           )
                      }} = 1
              )
            AND lower(Action) like "%dividend%"
        """
        smoothie = blend(query=blendsql, db=db, ingredients={LLM}, verbose=False)
        runtime = time.time() - start
        print(f"Completed with_blendsql benchmark in {runtime} seconds")
        print(f"Passed {smoothie.meta.num_values_passed} total values to LLM")
        times.append(runtime)
    print(
        f"For {iterations} iterations, average runtime is {statistics.mean(times)} with stdev {statistics.stdev(times)}"
    )
