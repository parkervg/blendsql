from typing import List, Optional, Callable, Any
from io import BytesIO
from PIL import Image
from functools import partial

from blendsql.models import TransformersVisionModel, LocalModel
from blendsql.ingredients.utils import batch_run_map
from blendsql._program import Program
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._exceptions import IngredientException
from blendsql import _constants as CONST
from blendsql import generate


class VQAProgram(Program):
    def __call__(
        self,
        model: TransformersVisionModel,
        question: str,
        values: List[bytes],
        sep: str,
        max_tokens: Optional[int] = None,
        regex: Optional[Callable[[int], str]] = None,
        **kwargs,
    ):
        content = [
            {"type": "text", "text": question},
        ] + [{"type": "image"} for _ in range(len(values))]
        if len(values) > 1:
            content.insert(
                0,
                [
                    {
                        "type": "text",
                        "text": f"Answer the below question for each provided image, with individual answers seperated by {sep}",
                    }
                ],
            )
        conversation = [
            {"role": "user", "content": content},
        ]
        prompt = model.processor.apply_chat_template(conversation)
        images: List[Image] = [Image.open(BytesIO(value)) for value in values]
        if isinstance(model, LocalModel) and regex is not None:
            response = generate.regex(
                model, prompt=prompt, media=images, regex=regex(len(values))
            )
        else:
            response = generate.text(
                model, prompt=prompt, media=images, max_tokens=max_tokens
            )
        return (response, prompt)


class VQA(MapIngredient):
    def run(
        self,
        model: TransformersVisionModel,
        question: str,
        values: List[bytes],
        regex: Optional[Callable[[int], str]] = None,
        **kwargs,
    ):
        if model is None:
            raise IngredientException(
                "ImageCaption requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if not all(isinstance(value, bytes) for value in values):
            raise IngredientException(
                f"All values must be 'byte' type for ImageCaption!"
            )
        pred_func = partial(
            model.predict,
            program=VQAProgram,
            question=question,
            regex=regex,
            **kwargs,
        )
        split_results: List[Any] = batch_run_map(
            pred_func,
            values=values,
            batch_size=CONST.MAP_BATCH_SIZE,
            sep=CONST.DEFAULT_ANS_SEP,
            nan_answer=CONST.DEFAULT_NAN_ANS,
        )
        return split_results
