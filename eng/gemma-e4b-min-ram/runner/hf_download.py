"""Small adapter that avoids passing empty HF tokens as Bearer headers."""

import os


def download_with_optional_token(hf_hub_download, **kwargs):
    token = os.environ.get("HF_TOKEN") or None
    return hf_hub_download(**kwargs, token=token)
