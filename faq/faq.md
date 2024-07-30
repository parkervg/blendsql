# FAQ
#### How does BlendSQL execute a query?
> BlendSQL handles traversal of the SQL AST and creation of temporary tables to execute a given query. 
> This allows BlendSQL to be DBMS-agnostic, and extendable into both SQLite, PostgreSQL, and other DBMS.

#### Why not just implement BlendSQL as a [user-defined function in SQLite](https://www.sqlite.org/c3ref/c_deterministic.html#sqlitedeterministic)?
> LLMs are expensive, both in terms of $ cost and compute time. When applying them to SQLite databases, we want to take special care in ensuring we're not applying them to contexts where they're not required. 
> This is [not easily achievable with UDFs](https://sqlite.org/forum/info/649ad4c62fd4b4e8cb5d6407107b8c8a9a0afaaf95a87805e5a8403a79e6616c), even when marked as a [deterministic function](https://www.sqlite.org/c3ref/c_deterministic.html#sqlitedeterministic).
> 
> BlendSQL is specifically designed to enforce an order-of-operations that 1) prioritizes vanilla SQL operations first, and 2) caches results from LLM ingredients so they don't need to be recomputed.
> For example:
> ```sql 
> SELECT {{LLMMap('What state is this NBA team from?', 'w::team')} FROM w 
>    WHERE num_championships > 3 
>    ORDER BY {{LLMMap('What state is this NBA team from?', 'w::team')}
> 
> ```
> BlendSQL makes sure to only pass those `team` values from rows which satisfy the condition `num_championship > 3` to the LLM. Additionally, since we assume the function is deterministic, we make a single LLM call and cache the results, despite the ingredient function being used twice.
#### So I get how to write BlendSQL queries. But why would I use this over vanilla SQLite? 
> Certain ingredients, like [LLMJoin](reference/ingredients/LLMJoin.md), will likely give seasoned SQL experts a headache at first. However, BlendSQL's real strength comes from it's use as an *intermediate representation for reasoning over structured + unstructured with LLMs*. Some examples of this can be found [here](examples/hybridqa.md).
