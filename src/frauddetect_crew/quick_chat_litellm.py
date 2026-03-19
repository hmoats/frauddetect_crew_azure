# quick_chat_litellm.py
import os, litellm
resp = litellm.completion(
    model=f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}",  # "azure/gpt-4.1"
    messages=[{"role":"user","content":"Hello from Azure via LiteLLM"}]
)
print(resp.choices[0].message["content"])
