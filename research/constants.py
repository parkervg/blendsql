from enum import Enum, auto


class EvalField(str, Enum):
    QUESTION = auto()
    GOLD_ANSWER = auto()
    PREDICTION = auto()
    PRED_BLENDSQL = auto()
    UID = auto()
    DB_PATH = auto()


SINGLE_TABLE_NAME = "w"
DOCS_TABLE_NAME = "documents"
CREATE_VIRTUAL_TABLE_CMD = f"CREATE VIRTUAL TABLE {DOCS_TABLE_NAME} USING fts5(title, content, tokenize = 'trigram');"
