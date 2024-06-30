from typing import Union, List, Set, Dict

from ..utils import get_tablename_colname
from ..db import Database


def unpack_options(
    options: Union[List[str], None], aliases_to_tablenames: Dict[str, str], db: Database
):
    unpacked_options: Union[List[str], None] = options
    if options is not None:
        if not isinstance(options, list):
            try:
                tablename, colname = get_tablename_colname(options)
                tablename = aliases_to_tablenames.get(tablename, tablename)
                # Optionally materialize a CTE
                if tablename in db.lazy_tables:
                    unpacked_options = (
                        db.lazy_tables.pop(tablename)
                        .collect()[colname]
                        .unique()
                        .tolist()
                    )
                else:
                    unpacked_options = db.execute_to_list(
                        f'SELECT DISTINCT "{colname}" FROM "{tablename}"'
                    )
            except ValueError:
                unpacked_options = options.split(";")
        unpacked_options: Set[str] = set(unpacked_options)
    return unpacked_options
