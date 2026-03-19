# quick_embed_azure.py
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
deploy = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
print("Embed deployment:", deploy)  # expect: text-embedding-3-small
resp = client.embeddings.create(model=deploy, input="ping")
print("Vector dim:", len(resp.data[0].embedding))
