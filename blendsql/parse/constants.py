from sqlglot import exp

SUBQUERY_EXP = (exp.Select,)
MODIFIERS = (
    exp.Delete,
    exp.AlterColumn,
    exp.AlterTable,
    exp.Drop,
    exp.RenameTable,
    exp.Drop,
    exp.DropPartition,
)
