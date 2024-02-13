import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import json
import os
import sqlite3
from collections import defaultdict
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List

import datasets
from wikiextractor.extract import Extractor, ignoreTag, resetIgnoredTags
from research.constants import EvalField

_CITATION = """\
@article{aly2021feverous,
  title={FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information},
  author={Aly, Rami and Guo, Zhijiang and Schlichtkrull, Michael and Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Cocarascu, Oana and Mittal, Arpit},
  journal={arXiv preprint arXiv:2106.05707},
  year={2021}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the FEVEROUS.
"""

_HOMEPAGE = "https://fever.ai/dataset/feverous.html"

_LICENSE = ""

_URL = "https://fever.ai/download/feverous/"
_TRAINING_FILE = "feverous_train_challenges.jsonl"
_DEV_FILE = "feverous_dev_challenges.jsonl"
_DATABASE = "feverous-wiki-pages-db.zip"

_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    "database": f"{_URL}{_DATABASE}",
}

EVIDENCE_TYPES = ["sentence", "cell", "header_cell", "table_caption", "item"]

extractor = Extractor(0, "", [], "", "")


def clean_markup(markup, keep_links=False, ignore_headers=True):
    """
    Clean Wikimarkup to produce plaintext.

    :param keep_links: Set to True to keep internal and external links
    :param ignore_headers: if set to True, the output list will not contain
    headers, only

    Returns a list of paragraphs (unicode strings).
    """

    if not keep_links:
        ignoreTag("a")

    # returns a list of strings
    paragraphs = extractor.clean_text(markup)
    resetIgnoredTags()

    if ignore_headers:
        paragraphs = filter(lambda s: not s.startswith("## "), paragraphs)

    return " ".join(list(paragraphs))


def get_table_id(meta):
    """
    meta types:
    - table_caption_18
    - cell_18_1_1
    - header_cell_18_0_0
    """
    if meta.startswith("table_caption"):
        return meta.split("_")[-1]
    if meta.startswith("header_cell") or meta.startswith("cell"):
        return meta.split("_")[-3]


def get_list_id(meta):
    """ "
    meta types:
    - item_4_25
    """
    return meta.split("_")[1]


def set_first_row_as_header(df: pd.DataFrame):
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    return df


def format_nested_table_json(table_json: dict):
    """
    TODO: how to handle table 'type'?
    """
    # Create numpy array
    #   row_start     column_start
    #      0                0
    #      1                0
    tablename = None
    flattened_values = []
    start_indices_arr = []
    span_indices_arr = []
    for row in table_json["table"]:
        start_indices_arr.extend(
            [list(map(lambda x: int(x), item["id"].split("_")[-2:])) for item in row]
        )
        span_indices_arr.extend(
            [[int(item["column_span"]) - 1, int(item["row_span"]) - 1] for item in row]
        )
        flattened_values.extend([clean_markup(item["value"]) for item in row])
    start_indices_arr, span_indices_arr, flattened_values = (
        np.array(start_indices_arr),
        np.array(span_indices_arr),
        np.array(flattened_values),
    )
    n_rows = start_indices_arr[:, 0].max()

    naive_df_data = []
    to_add_queue = []
    for row in range(n_rows):
        row_entry = [to_add_queue.pop()] if len(to_add_queue) > 0 else []
        indices = np.where(start_indices_arr[:, 0] == row)[0]
        spans = span_indices_arr[indices]
        values = flattened_values[indices]
        for _pos in range(indices.shape[0]):
            for _ in range(spans[_pos][0] + 1):
                row_entry.append(values[_pos])
            for _idx in range(spans[_pos][1]):
                to_add_queue.append(values[_pos])
        naive_df_data.append(row_entry)
    naive_df = pd.DataFrame(naive_df_data)
    naive_df = naive_df.replace("", np.nan)
    naive_df = naive_df.ffill()
    naive_df = naive_df.fillna("")
    if len(naive_df.columns) == 2:
        # Transpose, so LLM gets whole `attribute` context
        # naive_df.columns = ["attribute", "value"]
        naive_df = naive_df.T
    try:
        return set_first_row_as_header(naive_df)
    except:
        return naive_df

    # Simplest case: if less than 3 cells span multiple indices
    # But, if it has only 2 columns, use 'attribute', 'value' formatting
    is_simple_table = span_indices_arr[span_indices_arr > 0].shape[0] < 3
    if is_simple_table:
        if len(naive_df.columns) == 2:
            naive_df.columns = ["attribute", "value"]
            return (tablename, naive_df)
        try:
            return (tablename, set_first_row_as_header(naive_df))
        except:
            return (tablename, naive_df)
    try:
        reformatted_df = {}
        handled_rows = set()
        for idx, row in naive_df.iterrows():
            if idx in handled_rows:
                continue
            handled = False
            values_as_set = set(row.values)
            if len(values_as_set) == 1:
                # This should be tablename
                tablename = values_as_set.pop()
                continue
            for i in range(row.values.shape[0]):
                if handled or i == row.values.shape[0] - 1:
                    break
                _values = list(dict.fromkeys(row.values[i:].tolist()))
                # Check if they have any words in common
                tokenized_overlapping_values = [i.split(" ") for i in _values]
                tokens_in_common: set = set.intersection(
                    *map(set, tokenized_overlapping_values)
                )
                if len(tokens_in_common) > 0:
                    # We have some tokens in common
                    # Only get difference, and assign as column/values
                    columnname = " ".join(tokens_in_common)
                    values = [
                        " ".join([tok for tok in item if tok not in tokens_in_common])
                        for item in tokenized_overlapping_values
                    ]
                    reformatted_df[columnname] = values
                    handled = True
            if not handled:
                # Check if values are repeated even number of times
                # E.g. ['Number', 'Percent', 'Number', 'Percent']
                values_counter = Counter(row.values)
                duplicate_values = {(k, v) for k, v in values_counter.items() if v > 1}
                if len(duplicate_values) > 1:
                    evenly_duplicated_values = [i[0] for i in duplicate_values]
                    num_duplications = [i[1] for i in duplicate_values][0]
                    subtable = pd.DataFrame(naive_df.iloc[idx:, :])
                    handled_rows.update(range(idx, len(naive_df)))
                    subtable = set_first_row_as_header(subtable)
                    seen_columns = set()
                    for columnname in subtable.columns:
                        if columnname in seen_columns:
                            continue
                        if columnname in evenly_duplicated_values:
                            if columnname not in reformatted_df:
                                reformatted_df[columnname] = []
                            for _, row in subtable[columnname].T.iterrows():
                                reformatted_df[columnname].extend(row.values.tolist())
                        else:
                            # Make this a new column too
                            reformatted_df[columnname] = [
                                i
                                for i in subtable[columnname].tolist()
                                if i != columnname
                            ] * num_duplications
                        seen_columns.add(columnname)
                    handled = True
        max_v = max(len(v) for v in reformatted_df.values())
        for k, values in reformatted_df.items():
            if len(values) != max_v:
                assert max_v % len(values) == 0
                mult = max_v // len(values)
                multiplied_values = [
                    x for xs in [[v] * mult for v in values] for x in xs
                ]
                reformatted_df[k] = multiplied_values
        return (tablename, pd.DataFrame(reformatted_df))
    except:
        try:
            if len(set(naive_df.iloc[0].values.tolist())) == 1:
                tablename = naive_df.iloc[0].values[0]
                naive_df = set_first_row_as_header(naive_df.iloc[1:, :])
                if len(set(naive_df.columns)) == 1 and len(naive_df.columns) == 2:
                    tablename = f"{tablename} - {naive_df.columns[0]}"
                    naive_df.columns = ["Attribute", "Value"]
        except:
            pass
        return (tablename, naive_df)


def retrieve_context(example, cur):
    pages = {}
    evidences = []
    # Collect all page
    """
      meta types:
      - table_caption_18
      - cell_18_1_1
      - header_cell_18_0_0
      - sentence_0
      - item_4_25
      """
    tables = []
    for evidence in example["evidence"][:1]:
        content = evidence["content"]
        for item in content:
            # Example: 'Michael Folivi_header_cell_1_0_0'
            # page_id = Michael Folivi
            # meta = header_cell_1_0_0
            page_id, meta = item.split("_", 1)
            if page_id not in pages:
                data = cur.execute(
                    """
        SELECT data FROM wiki WHERE id = "{}"
        """.format(
                        page_id
                    )
                )
                for item in data.fetchall():
                    pages[page_id] = json.loads(item[0])
            if (
                meta.startswith("table_caption")
                or meta.startswith("cell")
                or meta.startswith("header_cell")
            ):
                table_id = get_table_id(meta)
                if table_id in tables:
                    continue
                else:
                    tables.append(table_id)
                table_json = pages[page_id]["table_{}".format(table_id)]
                evidences.append({"table": table_json, "tablename": page_id})
            elif meta.startswith("item"):
                list_id = get_list_id(meta)
                context = None
                for item in pages[page_id]["list_{}".format(list_id)]["list"]:
                    if item["id"] == meta:
                        context = item["value"]
                if context is not None:
                    evidences.append(
                        {"content": clean_markup(context), "title": page_id}
                    )
            else:
                context = pages[page_id][meta]
                evidences.append({"content": clean_markup(context), "title": page_id})

    table_list, context_list = [], []
    title_to_content: Dict[str, List[str]] = {}
    for evidence in evidences:
        if "table" in evidence:
            df = format_nested_table_json(evidence["table"])
            df_dict = df.to_dict(orient="split")
            table_list.append(
                {
                    "header": df_dict["columns"],
                    "rows": df_dict["data"],
                    "table_description": evidence["tablename"],
                }
            )
        else:
            if evidence["title"] not in title_to_content:
                title_to_content[evidence["title"]] = []
            title_to_content[evidence["title"]].append(evidence["content"])
        context_list.extend(
            [{"title": k, "content": " ".join(v)} for k, v in title_to_content.items()]
        )
    # Remove overlaps
    filtered_context_list = []
    context_list_titles = [item["title"] for item in context_list]
    for title in set(context_list_titles):
        content_candidates = []
        for item in context_list:
            if item["title"] == title:
                content_candidates.append(item["content"])
        chosen_content = sorted(content_candidates, key=len, reverse=True)[0]
        filtered_context_list.append({"title": title, "content": chosen_content})
    return table_list, filtered_context_list


def is_table_involved(example):
    # Check if the example is involving table.
    # We only consider the first evidence
    for evidence in example["evidence"][:1]:  # list
        is_valid = False
        content = evidence["content"]
        evidence_type_count = defaultdict(int)
        for item in content:
            page_id, meta = item.split("_", 1)
            for evidence_type in EVIDENCE_TYPES:
                if meta.startswith(evidence_type):
                    evidence_type_count[evidence_type] += 1
        for evidence_type in evidence_type_count:
            if evidence_type in ["cell", "header_cell", "table_caption"]:
                is_valid = True
        if is_valid:
            return True
    return False


class FEVEROUS(datasets.GeneratorBasedBuilder):
    """The FEVEROUS dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    EvalField.UID: datasets.Value("string"),
                    EvalField.QUESTION: datasets.Value("string"),
                    "table": datasets.features.Sequence(
                        {
                            "header": datasets.features.Sequence(
                                datasets.Value("string")
                            ),
                            "rows": datasets.features.Sequence(
                                datasets.features.Sequence(datasets.Value("string"))
                            ),
                            "table_description": datasets.Value("string"),
                        }
                    ),
                    "context": datasets.features.Sequence(
                        {
                            "title": datasets.Value("string"),
                            "content": datasets.Value("string"),
                        }
                    ),
                    EvalField.GOLD_ANSWER: datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "database": os.path.join(
                        downloaded_files["database"], "feverous_wikiv1.db"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "database": os.path.join(
                        downloaded_files["database"], "feverous_wikiv1.db"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath, database):
        con = sqlite3.connect(database)
        cur = con.cursor()
        with open(filepath, "r") as f:
            count = -1
            for _idx, line in enumerate(f):
                example = json.loads(line)
                statement = example["claim"]
                label = example["label"]
                # possible label: "NOT ENOUGH INFO", "REFUTES", "SUPPORTS"
                if is_table_involved(example):
                    # Retrieve related context from database
                    tables, contexts = retrieve_context(example, cur)
                    count += 1
                    yield count, {
                        EvalField.UID: str(example["id"]),
                        EvalField.QUESTION: statement,
                        "table": tables,
                        "context": contexts,
                        EvalField.GOLD_ANSWER: label,
                    }
