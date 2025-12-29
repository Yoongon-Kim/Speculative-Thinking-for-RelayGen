import os
cache_dir = None
project_dir = '/workspace/speculative_thinking'
hug_token = None
if cache_dir is not None:
    os.environ["HF_HOME"] = cache_dir

if hug_token is not None:
    from huggingface_hub import login
    login(token=hug_token)

api_keys = None