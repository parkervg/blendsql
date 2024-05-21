from typing import Iterable, Set
import re
from ..constants import DOCS_TABLE_NAME
from blendsql.db.utils import double_quote_escape


def to_serialized(
    db: "SQLite",
    ignore_tables: Iterable[str] = None,
    use_tables: Set[str] = None,
    num_rows: int = 0,
    tablename_to_description: dict = None,
    whole_table: bool = False,
    truncate_content: int = None,
) -> str:
    if all(x is not None for x in [ignore_tables, use_tables]):
        raise ValueError("Both `ignore_tables` and `use_tables` cannot be passed!")
    if ignore_tables is None:
        ignore_tables = set()
    serialized_db = []
    if use_tables:
        _create_clause_iter = [db.create_clause(tablename) for tablename in use_tables]
    else:
        _create_clause_iter = db.create_clauses()
    for tablename, create_clause in _create_clause_iter:
        # Check if it's an artifact of virtual table
        if re.search(r"^{}_".format(DOCS_TABLE_NAME), tablename):
            continue
        if tablename in ignore_tables:
            continue
        if use_tables is not None and tablename not in use_tables:
            continue
        if tablename_to_description is not None:
            if tablename in tablename_to_description:
                if tablename_to_description[tablename] is not None:
                    serialized_db.append(
                        f"Table Description: {tablename_to_description[tablename]}"
                    )
        if not whole_table:
            serialized_db.append(create_clause)
        if (num_rows > 0 and not tablename.startswith(DOCS_TABLE_NAME)) or whole_table:
            get_rows_query = (
                f'SELECT * FROM "{double_quote_escape(tablename)}" LIMIT {num_rows}'
                if not whole_table
                else f'SELECT * FROM "{double_quote_escape(tablename)}"'
            )
            serialized_db.append("/*")
            if whole_table:
                serialized_db.append("Entire table:")
            else:
                serialized_db.append(f"{num_rows} example rows:")
            serialized_db.append(f"{get_rows_query}")
            rows = db.execute_to_df(get_rows_query)
            if truncate_content is not None:
                # Truncate long strings
                rows = rows.map(
                    lambda x: (
                        f"{str(x)[:truncate_content]}..."
                        if isinstance(x, str) and len(str(x)) > truncate_content
                        else x
                    )
                )
            serialized_db.append(f"{rows.to_string(index=False)}")
            serialized_db.append("*/\n")
    serialized_db = "\n".join(serialized_db).strip()
    return serialized_db
