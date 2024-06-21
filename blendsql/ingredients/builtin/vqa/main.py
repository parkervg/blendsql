from typing import List

from blendsql.models import Model
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._exceptions import IngredientException


class ImageCaption(MapIngredient):
    def run(self, values: List[bytes], **kwargs):
        """Generates a caption for all byte images passed to it."""
        if not all(isinstance(value, bytes) for value in values):
            raise IngredientException(f"All values must be 'byte' type for LlavaVQA!")
        return self.model.predict(img_bytes=values)

    @staticmethod
    def predict(model: Model, img_bytes: List[bytes]):
        model_output = model.logits_generator(
            images=[Image.open(BytesIO(value)) for value in img_bytes],
            # prompt=prompt,
            generate_kwargs={"max_new_tokens": 200},
        )
        return [
            output[0]["generated_text"].lstrip(prompt).strip()
            for output in model_output
        ]
