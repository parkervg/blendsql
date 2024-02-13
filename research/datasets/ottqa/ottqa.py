import os
import json
import nltk
import datasets
from pathlib import Path

from research.constants import EvalField

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{chen2020open,
  title={Open question answering over tables and text},
  author={Chen, Wenhu and Chang, Ming-Wei and Schlinger, Eva and Wang, William and Cohen, William W},
  journal={arXiv preprint arXiv:2010.10439},
  year={2020}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the OTT-QA.
"""

_HOMEPAGE = "https://ott-qa.github.io"

_LICENSE = "MIT License"

_URL = "https://github.com/wenhuchen/OTT-QA/raw/a14ec408b2c22e24a44622b01e4242d95b7ecf08/released_data/"
_TRAINING_FILE = "train.traced.json"
_DEV_FILE = "dev.traced.json"

_URLS = {
    "tables": "https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json",
    "passages": "https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json",
}

WINDOW_SIZE = 3


class OTTQA(datasets.GeneratorBasedBuilder):
    """The OTTQA dataset"""

    def __init__(
        self,
        *args,
        db_output_dir: str,
        writer_batch_size=None,
        ottqa_dataset_url=_URL,
        **kwargs,
    ) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

        self._url = ottqa_dataset_url
        self.db_output_dir = Path(db_output_dir)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    EvalField.UID: datasets.Value("string"),
                    EvalField.DB_PATH: datasets.Value("string"),
                    EvalField.QUESTION: datasets.Value("string"),
                    "table_id": datasets.Value("string"),
                    "table": {
                        "header": datasets.features.Sequence(datasets.Value("string")),
                        "rows": datasets.features.Sequence(
                            datasets.features.Sequence(datasets.Value("string"))
                        ),
                    },
                    "passage": datasets.Value("string"),
                    "context": datasets.Value("string"),
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
        data_dir = dl_manager.download_and_extract(self._url)
        train_filepath = os.path.join(data_dir, "train.traced.json")
        dev_filepath = os.path.join(data_dir, "dev.traced.json")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_filepath,
                    "tablepath": downloaded_files["tables"],
                    "passagepath": downloaded_files["passages"],
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_filepath,
                    "tablepath": downloaded_files["tables"],
                    "passagepath": downloaded_files["passages"],
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath, tablepath, passagepath, data_dir):
        """Yields examples."""
        # data_id, question, table_id, gold_result_str

        with open(tablepath, encoding="utf-8") as f:
            tables = json.load(f)
        with open(passagepath, encoding="utf-8") as f:
            passages = json.load(f)

        # Format to database file
        if not self.db_output_dir.is_dir():
            self.db_output_dir.mkdir(parents=True)

        # dataset_split = Path(filepath).stem.split(".")[0]
        # output_db_filepath = None
        # output_db_filepath = self.db_output_dir / "ottqa.db"
        # add_tables = False
        # add_documents = False
        # if dataset_split == "train":
        #     db_filename = f"ottqa.db"
        #     output_db_filepath = self.db_output_dir / db_filename
        #     # if not output_db_filepath.is_file():
        #     if add_tables:
        #         logger.info(f"\nConstructing {db_filename} in {data_dir}...")
        #         tablename_to_table_json = {}
        #         tablename_to_unique_idx = {}
        #         for _table_id, table_data in tqdm(
        #             tables.items(), total=len(tables), desc="Formatting tables..."
        #         ):
        #             _tablename = table_data["title"]
        #             if _tablename not in tablename_to_unique_idx:
        #                 tablename_to_unique_idx[_tablename] = 0
        #
        #             tablename = f"{_tablename} ({tablename_to_unique_idx[_tablename]})"
        #             tablename_to_table_json[tablename] = {
        #                 "header": table_data["header"],
        #                 "rows": table_data["data"],
        #             }
        #             tablename_to_unique_idx[_tablename] += 1
        #
        #         csv_output_dir = self.db_output_dir / "csv"
        #         if not csv_output_dir.is_dir():
        #             csv_output_dir.mkdir(parents=True)
        #
        #         for tablename, table_json in tqdm(
        #             tablename_to_table_json.items(),
        #             total=len(tablename_to_table_json),
        #             desc="Saving tables to csv...",
        #         ):
        #             csv_save_path = (
        #                 self.db_output_dir
        #                 / "csv"
        #                 / f"{tablename.replace('/', ' ')}.csv"
        #             )
        #             if csv_save_path.is_file():
        #                 continue
        #             df = prepare_df_for_neuraldb_from_table(
        #                 table_json, add_row_id=False
        #             )
        #             df.to_csv(csv_save_path, index=False)
        #
        #     # Use csvs-to-sqlite to create many sqlite tables from our csvs
        #     # https://github.com/simonw/csvs-to-sqlite
        #     # error_bad_lines deprecated: https://github.com/simonw/csvs-to-sqlite/issues/88
        #     # need to run `pip install 'pandas==1.4.0'`
        #     # os.system("csvs-to-sqlite ")
        #
        #     # try:
        #     #     df.to_sql(
        #     #         tablename, sqlite_conn, if_exists="fail"
        #     #     )
        #     # except ValueError:
        #     #     logger.info(f"FAILED ON TABLE {tablename}")
        #
        #     if add_documents:
        #         documents_table_json = {"header": ["title", "content"], "rows": []}
        #         for doc_id, content in tqdm(
        #             passages.items(),
        #             total=len(passages),
        #             desc="Formatting documents...",
        #         ):
        #             title = doc_id.split("/")[-1].replace("_", " ")
        #             documents_table_json["rows"].append([title, content])
        #
        #         # Put into database
        #         sqlite_conn = sqlite3.connect(
        #             str(output_db_filepath), check_same_thread=True
        #         )
        #
        #         chunksize = 10000
        #
        #         def chunker(seq, size):
        #             return (seq[pos : pos + size] for pos in range(0, len(seq), size))
        #
        #         documents_df = pd.DataFrame(
        #             data=documents_table_json["rows"],
        #             columns=documents_table_json["header"],
        #         )
        #         c = sqlite_conn.cursor()
        #         c.execute(CREATE_VIRTUAL_TABLE_CMD)
        #         c.close()
        #         with tqdm(
        #             total=len(documents_df), desc="Uploading documents to db..."
        #         ) as pbar:
        #             for _i, cdf in enumerate(chunker(documents_df, chunksize)):
        #                 cdf.to_sql(
        #                     DOCS_TABLE_NAME,
        #                     sqlite_conn,
        #                     method="multi",
        #                     if_exists="append",
        #                     index=False,
        #                 )
        #                 pbar.update(chunksize)
        #
        #         sqlite_conn.close()
        #
        #         logger.info("\nFinished.")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, example in enumerate(data):
                table = tables[example["table_id"]]
                answer_node = example["answer-node"]
                answer = example["answer-text"]
                header, data, passage_context_str = self.construct_expanded_table(
                    table, passages, answer, answer_node
                )
                yield idx, {
                    EvalField.UID: example["question_id"],
                    EvalField.DB_PATH: str(self.db_output_dir / "ottqa.db"),
                    EvalField.QUESTION: example["question"],
                    "table_id": example["table_id"],
                    "table": {"header": header, "rows": data},
                    "passage": passage_context_str,
                    "context": table["title"]
                    + " | "
                    + table["section_title"]
                    + " | "
                    + table["section_text"]
                    + " | "
                    + table["intro"],
                    EvalField.GOLD_ANSWER: example["answer-text"],
                }

    def construct_expanded_table(self, table, passages, answer, answer_nodes):
        def process_link(link):
            return link.split("/")[-1].replace("_", " ")

        selected_passage = {}
        for answer_node in answer_nodes:
            link = answer_node[2]
            type_ = answer_node[3]
            if type_ == "passage":
                # Get passage and locate the sentence of answer
                passage_text = passages[link]
                sents = nltk.sent_tokenize(passage_text)
                has_answer_sent_idx = -1
                for idx, sent in enumerate(sents):
                    if " " + answer.lower() + " " in " " + sent.lower() + " ":
                        has_answer_sent_idx = idx
                selected_sents = sents[
                    max(0, has_answer_sent_idx - (WINDOW_SIZE - 1) // 2) : min(
                        len(sents) - 1, has_answer_sent_idx + (WINDOW_SIZE - 1) // 2
                    )
                ]
                selected_passage[process_link(link)] = " ".join(selected_sents)
            else:
                pass
        # linearize selected passgae
        passage_context_str = "passages: "
        for key in selected_passage:
            passage_context_str += "{}: {} | ".format(key, selected_passage[key])
        return table["header"], table["data"], passage_context_str
