---
hide:
  - toc
---

!!! warning

    Documentation here is a work in progress 


# Translating Natural Language to BlendSQL 

## nl_to_blendsql

::: blendsql.nl_to_blendsql.nl_to_blendsql.nl_to_blendsql
    handler: python
    show_source: false

## NLtoBlendSQLArgs

::: blendsql.nl_to_blendsql.args.NLtoBlendSQLArgs
    handler: python
    show_source: false

### Grammar-Based Correction

If you use the grammar correction feature of BlendSQL, please cite the original grammar prompting paper below.

```bibtex
@article{wang2024grammar,
  title={Grammar prompting for domain-specific language generation with large language models},
  author={Wang, Bailin and Wang, Zi and Wang, Xuezhi and Cao, Yuan and A Saurous, Rif and Kim, Yoon},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## FewShot

::: blendsql.prompts._prompts.FewShot
    handler: python
    show_source: false

## Examples 

::: blendsql.prompts._prompts.Examples
    handler: python
    show_source: false