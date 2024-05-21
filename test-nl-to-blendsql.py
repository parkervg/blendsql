from blendsql.nl_to_blendsql import nl_to_blendsql, NLtoBlendSQLArgs
from blendsql.models import TransformersLLM
from blendsql import LLMMap, LLMQA
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.prompts import FewShot
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    db = SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )
    # model = OllamaLLM("phi3")
    model = TransformersLLM("Qwen/Qwen1.5-0.5B")
    # model = OpenaiLLM("davinci-002")
    ingredients = {LLMMap, LLMQA}
    filtered_few_shot = FewShot.hybridqa.filter(ingredients)
    blendsql = nl_to_blendsql(
        "What was the result of the game played 120 miles west of Sydney?",
        db=db,
        model=model,
        ingredients=ingredients,
        few_shot_examples=filtered_few_shot,
        verbose=True,
        args=NLtoBlendSQLArgs(
            max_grammar_corrections=5,
            use_tables=["w"],
            include_db_content_tables=["w"],
            num_serialized_rows=3,
            use_bridge_encoder=True,
        ),
    )
