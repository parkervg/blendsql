# blend cli

```
usage: blendsql [-h] [-v]
                [db_url] [{openai,azure_openai,llama_cpp,transformers}]
                [model_name_or_path]

positional arguments:
  db_url                Database URL,
  {openai,azure_openai,llama_cpp,transformers}
                        Model type, for the Blender to use in executing the BlendSQL
                        query.
  model_name_or_path    Model identifier to pass to the selected model_type class.

optional arguments:
  -h, --help            show this help message and exit
  -v                    Flag to run in verbose mode.
```

Example Usage:

```bash
blendsql mydb.db openai gpt-3.5-turbo -v
```