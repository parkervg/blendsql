import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Callable

from blendsql.models import Model

TEST_QUESTION = "The quick brown fox jumps over the lazy dog"
TEST_FUNC = lambda x: x.lower().strip()

MODEL_A = "a"
MODEL_B = "b"


@dataclass
class DummyModelOutput:
    _variables: dict


class DummyModel(Model):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            tokenizer=None,
            **kwargs,
        )

    def _load_model(self):
        return self.model_name_or_path

    def generate(
        self, *args, funcs: Optional[Sequence[Callable]] = None, **kwargs
    ) -> List[str]:
        responses, key = None, None
        if self.caching:
            responses, key = self.check_cache(*args, **kwargs, funcs=funcs)
        if responses is None:
            responses = [str(uuid.uuid4())]
        if self.caching:
            self.cache[key] = responses
        return responses


def test_simple_cache():
    a = DummyModel(MODEL_A).generate(question=TEST_QUESTION)
    model_b = DummyModel(MODEL_A)
    b = model_b.generate(question=TEST_QUESTION)

    assert a == b
    assert model_b.num_generation_calls == 0


def test_different_models():
    a = DummyModel(MODEL_A).generate(question=TEST_QUESTION)
    b = DummyModel(MODEL_B).generate(question=TEST_QUESTION)

    assert a != b


def test_different_kwargs():
    a = DummyModel(MODEL_A).generate(question=TEST_QUESTION)
    b = DummyModel(MODEL_A).generate(question="This is a different question")

    assert a != b


def test_different_args():
    a = DummyModel(MODEL_A).generate("a", "b", "c", question=TEST_QUESTION)
    b = DummyModel(MODEL_A).generate("c", "d", "e", question=TEST_QUESTION)

    assert a != b


def test_same_funcs():
    a = DummyModel(MODEL_A).generate(question=TEST_QUESTION, funcs=[TEST_FUNC])
    model_b = DummyModel(MODEL_A)
    b = model_b.generate(question=TEST_QUESTION, funcs=[TEST_FUNC])

    assert a == b
    assert model_b.num_generation_calls == 0


def test_different_funcs():
    a = DummyModel(MODEL_A).generate(question=TEST_QUESTION, funcs=[TEST_FUNC])
    model_b = DummyModel(MODEL_A)
    b = model_b.generate(question=TEST_QUESTION, funcs=[lambda x: x + 1])

    assert a != b


def test_with_set_vars():
    a = DummyModel(MODEL_A).generate(question=TEST_QUESTION, random_set={"a", "b", "c"})

    b = DummyModel(MODEL_A).generate(question=TEST_QUESTION, random_set={"b", "c", "a"})

    assert a == b
