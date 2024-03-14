import logging
from typing import TypeVar

logger = logging.getLogger("minEarley")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

NO_VALUE = object()

T = TypeVar("T")
