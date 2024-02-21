# [FEVEROUS](https://fever.ai/dataset/feverous.html)

Here, we deal not with questions, but truth claims given a context of unstructured and structured data.

These claims should be judged as "SUPPORTS" or "REFUTES". Using BlendSQL, we can formulate this determination of truth as a function over facts. 

*Oyedaea is part of the family Asteraceae in the order Asterales.*
```sql
SELECT EXISTS (
    SELECT * FROM w0 WHERE "family:" = 'asteraceae' AND "order:" = 'asterales'
) 
```

*Sixty two year old Welsh journalist Jan Moir worked for a couple other papers before working at Daily Mail as an opinion columnist and has won several awards for her writing.*
```sql
SELECT (
    SELECT {{LLMMap('What age?', 'w0::born')}} = 62 FROM w0
) AND (
    {{
        LLMValidate(
            'Did Jan Moir work at a couple other papers before working at Daily Mail as an opinion columnist?',
            (SELECT * FROM documents)
        ) 
    }}
) AND (
    {{
        LLMValidate(
            'Has Jan Moir won several awards for her writing?',
            (SELECT * FROM documents)
        ) 
    }}
)
```

*Saunders College of Business, which is accredited by the Association to Advance Collegiate Schools of Business International, is one of the colleges of Rochester Institute of Technology established in 1910 and is currently under the supervision of Dean Jacqueline R. Mozrall.*
```sql
SELECT EXISTS(
    SELECT * FROM w0 
    WHERE "parent institution" = 'rochester institute of technology'
    AND "established" = '1910'
    AND "dean" = 'jacqueline r. mozrall'
) AND (
    {{
        LLMValidate(
            'Is Saunders College of Business (SCB) accredited by the Association to Advance Collegiate Schools of Business International (AACSB)?',
            (SELECT * FROM documents)
        )
    }}
)