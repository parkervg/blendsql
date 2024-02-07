from typing import Dict, Union
import re
from blendsql.ingredients.builtin.llm.utils import (
    construct_gen_clause,
)
from blendsql.ingredients.builtin.llm.endpoint import Endpoint
from blendsql import _programs as programs
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import QAIngredient, IngredientException
from blendsql.utils import get_tablename_colname
from blendsql.db.utils import single_quote_escape


class LLMQA(QAIngredient):
    def run(
        self,
        question: str,
        endpoint: Endpoint,
        long_answer: bool = False,
        value_limit: Union[int, None] = None,
        options: str = None,
        table_to_title: Dict[str, str] = None,
        **kwargs,
    ) -> Union[str, int, float]:
        if question is None:
            raise IngredientException("Need to specify `question` for LLMQA")
        # Unpack default kwargs
        subtable = self.unpack_default_kwargs(**kwargs)
        if value_limit is not None:
            subtable = subtable.iloc[:value_limit]
        if options is not None:
            try:
                tablename, colname = get_tablename_colname(options)
                tablename = kwargs.get("aliases_to_tablenames").get(tablename, tablename)
                options = (
                    self.db.execute_query(f'SELECT "{colname}" FROM "{tablename}"')[
                        colname
                    ]
                    .unique()
                    .tolist()
                )
            except ValueError:
                options = options.split(";")
            # Ensure options don't have a common prefix
            # This will be removed later
            # Related to this issue: https://github.com/guidance-ai/guidance/issues/232#issuecomment-1597125256
            options = [f"{idx}): {item}" for idx, item in enumerate(options)]
        gen_clause: str = construct_gen_clause(
            gen_type="select" if options else "gen",
            options=options,
            **endpoint.gen_kwargs,
        )
        program: str = (
            programs.QA_PROGRAM_CHAT(gen_clause)
            if endpoint.endpoint_name in CONST.OPENAI_CHAT_LLM
            else programs.QA_PROGRAM_COMPLETION(gen_clause)
        )
        res = endpoint.predict(
            program=program,
            options_dict=[{"option": item} for item in options] if options else None,
            question=question,
            serialized_db=subtable.to_string(),
            long_answer=long_answer,
            table_title=None,
        )
        if options is None:
            _result = res["result"]
        else:
            _result = re.sub(r"^\d+\):", "", res["result"])
        return "'{}'".format(single_quote_escape(_result.strip()))
