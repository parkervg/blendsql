from typing import List
from attr import attrs, attrib

from blendsql.models import Model
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._exceptions import IngredientException
from blendsql.ingredients.utils import partialclass


@attrs
class ImageCaption(MapIngredient):
    DESCRIPTION = """
    If we need to generate a caption for an image stored in the database, we can use the scalar function to map to a new column:
        `{{ImageCaption('table::column')}}`
    """
    model: Model = attrib(default=None)

    @classmethod
    def from_args(cls, model: Model = None):
        return partialclass(cls, model=model)

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
        from io import BytesIO
        from PIL import Image

        model_output = model.model_obj(
            images=[Image.open(BytesIO(value)) for value in values],
            # prompt=prompt,
            generate_kwargs={"max_new_tokens": 200},
        )
        model.num_generation_calls += 1
        return [output[0]["generated_text"].strip() for output in model_output]
