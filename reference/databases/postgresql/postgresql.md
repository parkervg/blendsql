---
hide:
  - toc
---
# PostreSQL 

!!! Installation

    You need to install the `psycopg2-binary` library to use this in blendsql.

::: blendsql.db._postgres.PostgreSQL
    handler: python
    show_source: true
    options:
      members:

## Creating a `blendsql` User

When executing a BlendSQL query, there are internal checks to ensure prior to execution that a given query does not contain any 'modify' actions.

However, it is still best practice when using PostgreSQL to create a dedicated 'blendsql' user with only the permissions needed. 

You can create a user with the required permissions with the script below (after invoking postgres via `psql`)

```bash
CREATE USER blendsql;
GRANT pg_read_all_data TO blendsql;
GRANT TEMP ON DATABASE mydb TO blendsql;
```

Now, we can initialize a PostgreSQL database with our new user.

```python
from blendsql.db import PostgreSQL
db = PostgreSQL("blendsql@localhost:5432/mydb")
```