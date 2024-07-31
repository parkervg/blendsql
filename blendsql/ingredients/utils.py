from typing import Union, List, Set, Dict, Any, Iterable
from functools import partial
from colorama import Fore
from tqdm import tqdm
from ast import literal_eval
import logging

from ..utils import get_tablename_colname
from ..db import Database
from .._logger import logger


def unpack_options(
    options: Union[List[str], str], aliases_to_tablenames: Dict[str, str], db: Database
) -> Set[str]:
    unpacked_options = options
    if not isinstance(options, list):
        try:
            tablename, colname = get_tablename_colname(options)
            tablename = aliases_to_tablenames.get(tablename, tablename)
            # Optionally materialize a CTE
            if tablename in db.lazy_tables:
                unpacked_options: list = (
                    db.lazy_tables.pop(tablename).collect()[colname].unique().tolist()
                )
            else:
                unpacked_options: list = db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{tablename}"'
                )
        except ValueError:
            unpacked_options = options.split(";")
    return set(unpacked_options)


def batch_run_map(
    f: partial, values: list, batch_size: int, sep: str, nan_answer: str
) -> List[Any]:
    split_results: List[Union[str, None]] = []
    # Only use tqdm if we're in debug mode
    context_manager: Iterable = (
        tqdm(
            range(0, len(values), batch_size),
            total=len(values) // batch_size,
            desc=f"Making calls to Model with batch_size {batch_size}",
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
        )
        if logger.level <= logging.DEBUG
        else range(0, len(values), batch_size)
    )

    for i in context_manager:
        answer_length = len(values[i : i + batch_size])
        max_tokens = answer_length * 15
        result = f(values=values[i : i + batch_size], sep=sep, max_tokens=max_tokens)
        # Post-process language model response
        _r = [i.strip() for i in result.strip(sep).split(sep)]
        # Try to map to booleans and `None`
        _r = [
            {
                "t": True,
                "f": False,
                "true": True,
                "false": False,
                "y": True,
                "n": False,
                "yes": True,
                "no": False,
                nan_answer: None,
            }.get(i.lower(), i)
            for i in _r
        ]
        expected_len = len(values[i : i + batch_size])
        if len(_r) != expected_len:
            logger.debug(
                Fore.YELLOW
                + f"Mismatch between length of values and answers!\nvalues:{expected_len}, answers:{len(_r)}"
                + Fore.RESET
            )
            logger.debug(_r)
        # Cut off, in case we over-predicted
        _r = _r[:expected_len]
        # Add, in case we under-predicted
        while len(_r) < expected_len:
            _r.append(None)
        split_results.extend(_r)
    for idx, i in enumerate(split_results):
        if i is None:
            continue
        if isinstance(i, str):
            i = i.replace(",", "")
        try:
            split_results[idx] = literal_eval(i)
            assert isinstance(i, (float, int, str))
        except (ValueError, SyntaxError, AssertionError):
            continue
    return split_results
