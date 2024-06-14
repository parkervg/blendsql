from blendsql.models import OllamaLLM, OpenaiLLM
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.nl_to_blendsql import nl_to_blendsql, NLtoBlendSQLArgs, FewShot
from blendsql import LLMMap, LLMQA, blend

if __name__ == "__main__":
    model = OllamaLLM("phi3")
    db = SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )
    prediction = nl_to_blendsql(
        question="Show me all info about the game played 120 miles west of Sydney",
        db=db,
        model=model,
        ingredients={LLMQA, LLMMap},
        correction_model=OpenaiLLM("gpt-3.5-turbo"),
        few_shot_examples=FewShot.hybridqa,
        args=NLtoBlendSQLArgs(
            use_tables=["w"], include_db_content_tables=["w"], use_bridge_encoder=True
        ),
        verbose=True,
    )
    smoothie = blend(
        query=prediction,
        blender=OpenaiLLM("gpt-3.5-turbo"),
        ingredients={LLMMap, LLMQA},
        db=db,
    )
    print(smoothie.df)
