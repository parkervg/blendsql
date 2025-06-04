from sqlglot import exp

SUBQUERY_EXP = (exp.Select,)
MODIFIERS = (
    exp.Delete,
    exp.AlterColumn,
    exp.AlterIndex,
    exp.AlterDistStyle,
    exp.AlterSortKey,
    exp.Alter,
    exp.Drop,
    exp.RenameColumn,
    exp.AlterRename,
    exp.Drop,
    exp.DropPartition,
)
