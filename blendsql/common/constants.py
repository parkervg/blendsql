import json

HF_REPO_ID = "parkervg/blendsql-test-dbs"

DEFAULT_ANS_SEP = ";"
DEFAULT_NAN_ANS = "-"
# DEFAULT_CONTEXT_FORMATTER = lambda df: json.dumps(
#     df.to_dict(orient="records"), ensure_ascii=False, indent=4
# )
DEFAULT_CONTEXT_FORMATTER = lambda df: json.dumps(
    df.to_dict(as_series=False), ensure_ascii=False, indent=4
)
INDENT = lambda n=1: "    " * n
