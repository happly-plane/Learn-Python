import openai

openai.api_key = "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"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="你是那个版本的模型",
    max_tokens=128,
    temperature=0.5,
)

completed_text = response["choices"][0]["text"]
print(completed_text)