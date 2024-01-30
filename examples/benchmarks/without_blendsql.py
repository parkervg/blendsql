import os
import time
import pandas as pd
import sqlite3
from blendsql import init_secrets
import openai
import statistics
import logging
from colorama import Fore
from typing import List, Union

init_secrets("secrets.json")


def construct_messages_payload(prompt: Union[str, None], question: str) -> List:
    messages = []
    # Add system prompt
    messages.append({"role": "system", "content": "" if prompt is None else prompt})
    """
    At this point we cannot provide the conversation history to Azure models
    so just passing the current user input as input to the model.
    """
    messages.append({"role": "user", "content": question})
    return messages


map_prompt = """Answer the question row-by-row, in order.
Values can be either '1' (True) or '0' (False).
The answer should be a list separataed by ';', and have {answer_length} items in total.

Question: {question}

{values}
"""

if __name__ == "__main__":
    db_path = "./tests/multi_table.db"
    con = sqlite3.connect(db_path)
    iterations = 10
    times = []
    openai.api_type = os.getenv("API_TYPE", "azure")
    openai.api_version = os.getenv("API_VERSION", "2023-03-15-preview")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    for _ in range(iterations):
        start = time.time()
        # Select initial query results
        sql = """
        SELECT * FROM constituents WHERE sector = 'Information Technology'
        """
        question = "does this company manufacture cell phones?"
        target_column = "Name"
        df = pd.read_sql(sql, con)

        # Make our calls to the LLM
        values = df[target_column].unique().tolist()
        # values_dict = [{"value": value, "idx": idx} for idx, value in enumerate(values)]
        split_results = []
        # Pass in batches
        batch_size = 20
        for i in range(0, len(values), batch_size):
            prompt = map_prompt.format(
                answer_length=len(values[i : i + batch_size]),
                question=question,
                values="\n".join(values[i : i + batch_size]),
            )
            # print(prompt)
            payload = construct_messages_payload(
                prompt=prompt, question="What's the total value of my portfolio?"
            )
            res = openai.ChatCompletion.create(
                engine="gpt-4",
                # Can be one of {"gpt-4", "gpt-4-32k", "gpt-35-turbo", "text-davinci-003"}, or others in Azure
                messages=payload,
                temperature=0.0,
            )["choices"][0]["message"]["content"]
            _r = [i.strip() for i in res.strip(";").split(";")]
            expected_len = len(values[i : i + batch_size])
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
        print(f"Completed without_blendsql benchmark in {runtime} seconds")
        print(f"Passed {values_passed} total values to LLM")
        times.append(runtime)
    print(
        f"For {iterations} iterations, average runtime is {statistics.mean(times)} with stdev {statistics.stdev(times)}"
    )
