from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # parser_model_type: str = field(
    #     metadata={
    #         "help": "Model type of the parser model. Accepted values are openai (default), hf and sagemaker"
    #     }
    # )
    # blender_model_type: str = field(
    #     metadata={
    #         "help": "Model type of the blender model. Accepted values are openai (default), hf and sagemaker"
    #     }
    # )
    parser_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    prompt_and_pray_model_type: str = field(
        default=None,
        metadata={
            "help": "Model type of the prompt and pray model. Accepted values are openai (default), hf and sagemaker"
        },
    )
    blender_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    prompt_and_pray_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    parser_temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature to use for parser"},
    )
    blender_temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature to use for blender"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
