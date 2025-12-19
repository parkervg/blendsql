import os
from pathlib import Path

from blendsql.common.logger import Color, logger

from src.config import MODEL_NAME_OR_PATH, FILENAME, LOCAL_GGUF_FILEPATH


def download_model_if_needed() -> bool:
    """
    Download GGUF model from Hugging Face if it doesn't exist locally.

    Returns:
        True if model is available (already exists or downloaded successfully)
    """
    if Path(LOCAL_GGUF_FILEPATH).is_file():
        logger.debug(Color.success(f"✓ Model already exists at {LOCAL_GGUF_FILEPATH}"))
        return True

    logger.debug(
        Color.update(
            f"Downloading model \n MODEL_NAME_OR_PATH={MODEL_NAME_OR_PATH} FILENAME={FILENAME}"
        )
    )

    # Ensure models directory exists
    if not LOCAL_GGUF_FILEPATH.parent.is_dir():
        LOCAL_GGUF_FILEPATH.parent.mkdir()

    print(
        f"""
        hf download \
        {MODEL_NAME_OR_PATH} \
        {FILENAME} \
        --local-dir {str(LOCAL_GGUF_FILEPATH.parent)} 
        """
    )
    # Download using huggingface-cli
    os.system(
        f"""
        hf download \
        {MODEL_NAME_OR_PATH} \
        {FILENAME} \
        --local-dir {str(LOCAL_GGUF_FILEPATH.parent)} 
        """
    )

    if Path(LOCAL_GGUF_FILEPATH).is_file():
        logger.debug(Color.success("✓ Model downloaded successfully"))
        return True
    else:
        logger.debug(Color.error("✗ Model download failed"))
        return False
