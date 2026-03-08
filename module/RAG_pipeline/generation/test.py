from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",  # vLLM local server does not require real OpenAI key
)

resp = client.chat.completions.create(
    model="BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM",
    messages=[{"role": "user", "content": "What is hypertension?"}],
    temperature=0.2,
)

print(resp.choices[0].message.content)