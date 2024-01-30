SINGLE_TABLE_NAME = "w"
DOCS_TABLE_NAME = "documents"
CREATE_VIRTUAL_TABLE_CMD = f"CREATE VIRTUAL TABLE {DOCS_TABLE_NAME} USING fts5(title, content, tokenize = 'trigram');"
