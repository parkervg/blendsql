from typing import List


from blendsql.ingredients.ingredient import MapIngredient
from blendsql._exceptions import IngredientException


class VQA(MapIngredient):
    def run(self, question: str, values: List[bytes], **kwargs):
        """Given a list of byte arrays, calls a tiny Llava model
        to answer a given question.
        """
        if not all(isinstance(value, bytes) for value in values):
            raise IngredientException(f"All values must be 'byte' type for LlavaVQA!")
        model_output = self.model.predict(question=question, img_bytes=values)
        return model_output
