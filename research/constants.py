from dataclasses import dataclass


@dataclass
class EvalField:
    QUESTION = "question"
    GOLD_ANSWER = "gold_answer"
    PREDICTION = "prediction"
    PRED_BLENDSQL = "pred_blendsql"
    UID = "uid"
    DB_PATH = "db_path"


SINGLE_TABLE_NAME = "w"
DOCS_TABLE_NAME = "documents"
CREATE_VIRTUAL_TABLE_CMD = f"CREATE VIRTUAL TABLE {DOCS_TABLE_NAME} USING fts5(title, content, tokenize = 'trigram');"
