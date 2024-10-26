```sql
SELECT {{LLMMap('Total points?', 'w::score')}} AS Total, score FROM w 
WHERE Total > 12
```
