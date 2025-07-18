import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="https://litellm.aida4u.com"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-4o",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)