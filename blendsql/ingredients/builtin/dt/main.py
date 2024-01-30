from datetime import datetime

from .utils import _parse_date
from ...ingredient import StringIngredient


class DT(StringIngredient):
    def run(self, start: str = None, end: str = None, fmt="%Y-%m-%d", **kwargs) -> str:
        """Calls a Python function to convert natural language referents
        to relative dates into absolute datetime values.

        Args:
            start: str, optional NL string (e.g. 'last year')
            end: str, optional NL string (e.g. 'last week')
            fmt: str datetime.strftime format

        Returns:
            New SQL expression with datetime constraints
        """
        # Unpack default kwargs
        tablename, colname = self.unpack_default_kwargs(**kwargs)
        start_dt, end_dt = _parse_date(
            today_dt=datetime.strptime("2022-12-01", "%Y-%m-%d"), start=start, end=end
        )

        return f'"{tablename}"."{colname}" > \'{start_dt.strftime(fmt)}\' AND "{tablename}"."{colname}" < \'{end_dt.strftime(fmt)}\''
