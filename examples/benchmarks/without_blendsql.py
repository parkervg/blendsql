import time
import os
import pandas as pd
import sqlite3
from openai import AzureOpenAI
import statistics
import logging
from colorama import Fore
from typing import List, Union
from dotenv import load_dotenv

from blendsql._constants import VALUE_BATCH_SIZE
from blendsql.utils import fetch_from_hub
from constants import model


def construct_messages_payload(prompt: Union[str, None], question: str) -> List:
    messages = []
    # Add system prompt
    # messages.append({"role": "system", "content": "" if prompt is None else prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


map_prompt = """Answer the question row-by-row, in order.
Values can be either '1' (True) or '0' (False).
The answer should be a list separated by ';', and have {answer_length} items in total.

Question: {question}

{values}
"""


def openai_setup() -> None:
    """Setup helper for AzureOpenAI and OpenAI models."""
    if all(
        x is not None
        for x in {
            os.getenv("TENANT_ID"),
            os.getenv("CLIENT_ID"),
            os.getenv("CLIENT_SECRET"),
        }
    ):
        try:
            from azure.identity import ClientSecretCredential
        except ImportError:
            raise ValueError(
                "Found ['TENANT_ID', 'CLIENT_ID', 'CLIENT_SECRET'] in .env file, using Azure OpenAI\nIn order to use Azure OpenAI, run `pip install azure-identity`!"
            ) from None
        credential = ClientSecretCredential(
            tenant_id=os.environ["TENANT_ID"],
            client_id=os.environ["CLIENT_ID"],
            client_secret=os.environ["CLIENT_SECRET"],
            disable_instance_discovery=True,
        )
        access_token = credential.get_token(
            os.environ["TOKEN_SCOPE"],
            tenant_id=os.environ["TENANT_ID"],
        )
        os.environ["AZURE_OPENAI_API_KEY"] = access_token.token
    elif os.getenv("AZURE_OPENAI_API_KEY") is not None:
        pass
    else:
        raise ValueError(
            "Error authenticating with OpenAI\n Without explicit `OPENAI_API_KEY`, you need to provide ['TENANT_ID', 'CLIENT_ID', 'CLIENT_SECRET']"
        ) from None


if __name__ == "__main__":
    load_dotenv()
    openai_setup()
    con = sqlite3.connect(fetch_from_hub("multi_table.db"))
    iterations = 1
    times = []
    client = AzureOpenAI()
    print(f"Using {model}...")
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
        batch_size = VALUE_BATCH_SIZE
        for i in range(0, len(values), batch_size):
            prompt = map_prompt.format(
                answer_length=len(values[i : i + batch_size]),
                question=question,
                values="\n".join(values[i : i + batch_size]),
            )
            # print(prompt)
            payload = construct_messages_payload(prompt=prompt, question="")
            res = (
                client.chat.completions.create(
                    model=model,
                    # Can be one of {"gpt-4", "gpt-4-32k", "gpt-35-turbo", "text-davinci-003"}, or others in Azure
                    messages=payload,
                )
                .choices[0]
                .message.content
            )
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
