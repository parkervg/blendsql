import os
import statistics
import time
import pandas as pd
import sqlite3
from blendsql.ingredients.builtin.llm.openai_llm import OpenaiLLM
from blendsql import init_secrets
from tqdm import tqdm
import logging
from colorama import Fore

init_secrets("secrets.json")

if __name__ == "__main__":
    start = time.time()
    db_path = "./tests/multi_table.db"
    con = sqlite3.connect(db_path)
    iterations = 10
    times = []
    for _ in range(iterations):
        # Select initial query results
        sql = """
        SELECT * FROM constituents WHERE sector = 'Information Technology'
        """
        question = "does this company manufacture cell phones?"
        target_column = "Name"
        df = pd.read_sql(sql, con)

        # Make our calls to the LLM
        endpoint = OpenaiEndpoint(endpoint="gpt-4")
        values = df[target_column].unique().tolist()
        values_dict = [{"value": value, "idx": idx} for idx, value in enumerate(values)]
        split_results = []
        # Pass in batches
        batch_size = 20
        for i in tqdm(
            range(0, len(values_dict), batch_size),
            total=len(values_dict) // batch_size,
            desc=f"Making calls to LLM with batch_size {batch_size}",
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
        ):
            res = endpoint.predict(
                program=programs.MAP_PROGRAM_CHAT,
                chat=True,
                question=question,
                sep=";",
                answer_length=len(values_dict[i : i + batch_size]),
                values_dict=values_dict[i : i + batch_size],
                binary=1,
                example_outputs=None,
            )
            _r = [i.strip() for i in res["result"].strip(";").split(";")]
            expected_len = len(values_dict[i : i + batch_size])
            if len(_r) != expected_len:
                logging.debug(
                    Fore.YELLOW
                    + f"Mismatch between length of values and answers!\nvalues:{expected_len}, answers:{len(_r)}"
                    + Fore.RESET
                )
                logging.debug(_r)
            # Cut off, in case we over-predicted
            _r = _r[:expected_len]
            # Add, in case we under-predicted
            while len(_r) < expected_len:
                _r.append(None)
            split_results.extend(_r)
        values_passed = len(split_results)
        df_as_dict = {target_column: [], question: []}
        for idx, value in enumerate(values):
            df_as_dict[target_column].append(value)
            df_as_dict[question].append(
                split_results[idx] if len(split_results) - 1 >= idx else None
            )
        subtable = pd.DataFrame(df_as_dict)

        # Add new_table to original table
        new_table = df.merge(subtable, how="left", on=target_column)
        new_table.to_sql("modified_constituents", con, if_exists="replace", index=False)
        # Now, new table has original columns + column with the name of the question we answered
        sql = """
        SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)'
            FROM account_history
            WHERE Symbol IN
              (
                  SELECT Symbol FROM modified_constituents
                  WHERE sector = 'Information Technology'
                  AND "does this company manufacture cell phones?" = 1 
              )
            AND lower(Action) like "%dividend%"
        """
        answer = pd.read_sql(sql, con)
        con.execute(f"DROP TABLE 'modified_constituents'")
        runtime = time.time() - start
        print(
            f"Completed without_blendsql_with_guidance benchmark in {runtime} seconds"
        )
        print(f"Passed {values_passed} total values to LLM")
        times.append(runtime)
    print(
        f"For {iterations} iterations, average runtime is {statistics.mean(times)} with stdev {statistics.stdev(times)}"
    )
