import os


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Export a Hugging Face access token before running: export HF_TOKEN=hf_..."
        )
    return token
