Download the predictions from https://huggingface.co/kaixinm/CORE/blob/main/data/retriever_data.zip  here.

From the work https://github.com/Mayer123/UDT-QA

```
@inproceedings{ma-etal-2022-open-domain,
    title = "Open-domain Question Answering via Chain of Reasoning over Heterogeneous Knowledge",
    author = "Ma, Kaixin  and
      Cheng, Hao  and
      Liu, Xiaodong  and
      Nyberg, Eric  and
      Gao, Jianfeng",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.392",
    doi = "10.18653/v1/2022.findings-emnlp.392",
    pages = "5360--5374",
    abstract = "We propose a novel open-domain question answering (ODQA) framework for answering single/multi-hop questions across heterogeneous knowledge sources. The key novelty of our method is the introduction of the intermediary modules into the current retriever-reader pipeline. Unlike previous methods that solely rely on the retriever for gathering all evidence in isolation,our intermediary performs a chain of reasoning over the retrieved set. Specifically, our method links the retrieved evidence with its related global context into graphs and organizes them into a candidate list of evidence chains. Built upon pretrained language models, our system achieves competitive performance on two ODQA datasets, OTT-QA and NQ, against tables and passages from Wikipedia.In particular, our model substantially outperforms the previous state-of-the-art on OTT-QA with an exact match score of 47.3 (45{\%} relative gain).",
}
```