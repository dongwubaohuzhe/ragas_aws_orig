"""Model configuration for different Bedrock models.

All models use ChatBedrockConverse (Converse API), which accepts
standardized 'temperature' and 'max_tokens' parameters regardless
of the underlying model provider.
"""

SUPPORTED_MODELS = {
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "anthropic.claude-3-7-sonnet-20250219-v1:0": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "amazon.titan-text-express-v1": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "amazon.titan-text-lite-v1": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "amazon.nova-pro-v1:0": {
        "temperature": 0.1,
        "max_tokens": 4096,
    },
}

SUPPORTED_EMBEDDINGS = [
    "amazon.titan-embed-text-v1",
    "amazon.titan-embed-text-v2:0",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
]

def get_model_config(model_id):
    """Get configuration for a specific model"""
    return SUPPORTED_MODELS.get(model_id, {"temperature": 0.1, "max_tokens": 4096})