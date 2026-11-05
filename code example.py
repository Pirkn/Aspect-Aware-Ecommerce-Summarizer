from openai import OpenAI

client = OpenAI(
    api_key=RUNPOD_TOKEN,
    base_url="https://api.runpod.ai/v2/7ja9cfymaef7os/openai/v1",
)

response = client.chat.completions.create(
    model="berkhan/llama31-8b-turkish-sft-v3-merged",
    messages=[
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_CONTENT},
    ],
    temperature=0.0,
    max_tokens=128,
)

print(response.choices[0].message.content)