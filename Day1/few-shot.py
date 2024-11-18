from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": """
            Translate the following English word into Korean, using the examples below as a guide.

            Examples:
            Monkey = 원숭이
            Bear = 곰
            Giraffe = 기린
            Zebra = 얼룩말
            Panda = 팬더
            
            Word to translate: dog
            """
        }
    ]
)

print(completion.choices[0].message)
