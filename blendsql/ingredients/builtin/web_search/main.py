from attr import attrs, attrib
from pathlib import Path
from dotenv import load_dotenv
import os
import requests

from blendsql.ingredients.utils import partialclass
from blendsql.db.utils import single_quote_escape
from blendsql.ingredients.ingredient import QAIngredient


@attrs(kw_only=True)
class BingWebSearch(QAIngredient):
    DESCRIPTION = """
    """

    env: str = attrib(default=".")
    search_url: str = attrib(default="https://api.bing.microsoft.com/v7.0/search")
    k: int = attrib(default=2)

    azure_subscription_key: str = attrib(init=False)

    def __attrs_post_init__(self):
        _env = Path(self.env)
        env_filepath = _env / ".env" if _env.is_dir() else _env
        load_dotenv(str(env_filepath))
        assert (
            os.getenv("AZURE_SUBSCRIPTION_KEY", None) is not None
        ), "BingWebSearch ingredient needs a 'AZURE_SUBSCRIPTION_KEY' key in your .env config!"

    @classmethod
    def from_args(cls, k: int = 2):
        return partialclass(cls, k=k)

    def run(self, question: str, **kwargs):
        headers = {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_SUBSCRIPTION_KEY")}
        params = {
            "q": question,
            "textDecorations": False,
            # "textFormat": "HTML",
            "count": self.k,
            "responseFilter": ["Webpages"],
        }
        response = requests.get(self.search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        return "'{}'".format(
            single_quote_escape(
                "\n\n".join(
                    [
                        f"## DOCUMENT {idx+1}\n\n{j['snippet']}"
                        for idx, j in enumerate(search_results["webPages"]["value"])
                    ]
                ).strip()
            )
        )
