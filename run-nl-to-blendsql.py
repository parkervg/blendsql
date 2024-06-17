from blendsql.models import OllamaLLM, TransformersLLM
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.nl_to_blendsql import nl_to_blendsql, NLtoBlendSQLArgs, FewShot
from blendsql import LLMMap, LLMQA, blend

if __name__ == "__main__":
    ollama_model = OllamaLLM("phi3")
    transformers_model = TransformersLLM("Qwen/Qwen1.5-0.5B", caching=False)
    db = SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )
    while True:
        # question = "Show me all info about the game played 120 miles west of Sydney"
        question = input(">>> ")
        print("\n")
        prediction = nl_to_blendsql(
            question=question,
            db=db,
            model=ollama_model,
            ingredients={LLMQA, LLMMap},
            correction_model=transformers_model,
            few_shot_examples=FewShot.hybridqa,
            args=NLtoBlendSQLArgs(
                max_grammar_corrections=3,
                use_tables=["w"],
                include_db_content_tables=["w"],
                use_bridge_encoder=True,
            ),
            verbose=True,
        )
        smoothie = blend(
            query=prediction,
            blender=transformers_model,
            ingredients={LLMMap, LLMQA},
            db=db,
        )
        print(smoothie.df)
