from typing import Tuple


def load_from_hub(repo_id: str) -> Tuple[str, str]:
    """
    Download a model and a relation_map from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    # Get the model from the Hub, download and cache the model on your local disk
    downloaded_model_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt"
    )

    # Get the relation_map from the Hub
    downloaded_relation_map_file = hf_hub_download(
        repo_id=repo_id,
        filename="relation_map.json"
    )

    return downloaded_model_file, downloaded_relation_map_file
