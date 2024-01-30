from blendsql import LLMMap, LLMQA, LLMJoin, blend, init_secrets
from blendsql.db import SQLiteDBConnector

init_secrets("../secrets.json")

if __name__ == "__main__":
    #  blendsql = """
    # {{
    #  LLMQA(
    #      'What were the position held by William Warren Barbour during his life?',
    #      (
    #           SELECT * from w WHERE vacator like 'william warren barbour%';
    #      )
    #  )
    #  }}
    #  """
    #  db_path = "research/db/fetaqa/totto_source/train_json/example-0.db"

    # blendsql = """
    # {{
    #   LLMQA(
    #         'How is Dyro ranked in the year of 2014',
    #             (
    #                 SELECT category, result FROM w WHERE nominee = 'dyro' AND year = '2014'
    #             )
    #         )
    #  }}
    #  """
    # db_path = "research/db/fetaqa/totto_source/train_json/example-4880.db"

    # blendsql = """
    # {{
    #  LLMQA(
    #        'For what work does Andy Karl win his  Olivier Award?',
    #            (
    #                SELECT year, work FROM w WHERE award like '%Olivier Award';
    #            )
    #        )
    # }}
    # """
    # db_path = "research/db/fetaqa/totto_source/dev_json/example-2274.db"

    blendsql = """
       {{
        LLMQA(
              'In what films did Pooja Ramachandran play Cathy?',
                  (
                      SELECT film FROM w WHERE role='cathy';
                  )
              )
       }}
       """
    db_path = "research/db/fetaqa/totto_source/train_json/example-2954.db"

    db = SQLiteDBConnector(db_path)
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients={LLMMap, LLMQA, LLMJoin},
        overwrite_args={"use_endpoint": "gpt-4", "long_answer": True},
        infer_map_constraints=True,
        silence_db_exec_errors=False,
        verbose=True,
    )
    print("--------------------------------------------------")
    print("ANSWER:")
    print([i for i in smoothie.df.values.flat])
    print("--------------------------------------------------")
    print(smoothie.meta.example_map_outputs)
    print(smoothie.meta.num_values_passed)
