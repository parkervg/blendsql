from typing import List, Tuple
from io import BytesIO
from PIL import Image

from blendsql.models import Model
from blendsql._program import Program
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._exceptions import IngredientException


class ImageCaptionProgram(Program):
    def __call__(
        self, model: Model, img_bytes: List[bytes], **kwargs
    ) -> Tuple[List[str], str]:
        model_output = model.model_obj(
            images=[Image.open(BytesIO(value)) for value in img_bytes],
            # prompt=prompt,
            generate_kwargs={"max_new_tokens": 200},
        )
        return ([output[0]["generated_text"].strip() for output in model_output], "")


class ImageCaption(MapIngredient):
    DESCRIPTION = """
    If we need to generate a caption for an image stored in the database, we can use the scalar function to map to a new column:
        `{{ImageCaption('table::column')}}`
    """

    def run(self, model: Model, values: List[bytes], **kwargs):
        """Generates a caption for all byte images passed to it."""
        if model is None:
            raise IngredientException(
                "ImageCaption requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if not all(isinstance(value, bytes) for value in values):
            raise IngredientException(
                f"All values must be 'byte' type for ImageCaption!"
            )
        return model.predict(program=ImageCaptionProgram, img_bytes=values, **kwargs)
