from .parse.dialect import BlendSQLFunction
import sqlglot

setattr(sqlglot.exp, "BlendSQLFunction", BlendSQLFunction)

from .blendsql import BlendSQL
from . import configure as config
