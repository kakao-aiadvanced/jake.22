from openai import OpenAI
client = OpenAI()

# Simple
# prompt = """
# 아래 예시를 참고해서 다음 수식을 계산해줘.
#
# # 더하기 연산 예시
# Solve the following problem step-by-step: 23 + 47
# Step-by-step solution:
# 1. 일의 자리 수를 더합니다.
# 2. 일의 자리 수의 결과로 나온 십의 자리 수와 십의 자리 수를 더합니다.
# 3. 그 다음 자리 수를 계산합니다.
#
# # 빼기 연산 예시
# Solve the following problem step-by-step: 123 - 58
#
# Step-by-step solution:
# 1. 일의 자리 수부터 뺍니다.
# 2. 빼기 연산 중 빌려오기가 필요한 경우, 다음 자리 수에서 1을 빼고 10을 더합니다.
# 3. 1번과 2번 과정을 반복합니다.
#
# Solve the following problem step-by-step: 345 + 678 - 123
# """
#
# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": prompt,
#         }
#     ]
# )
#
# print(completion.choices[0].message.content)

# Intermediate - 1
# prompt = """
# Solve the following logic puzzle step-by-step:
# Three friends, Alice, Bob, and Carol, have different favorite colors: red, blue, and green. We know that:
# 1. Alice does not like red.
# 2. Bob does not like blue.
# 3. Carol likes green.
#
# Determine the favorite color of each friend.
#
# Step-by-step solution:
# 1. Alice 가 좋아하는 색깔을 찾아
# 2. Bob 가 좋아하는 색깔을 찾아
# 3. Carol 이 좋아하는 색깔을 찾아
# """
#
# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": prompt,
#         }
#     ]
# )
#
# print(completion.choices[0].message.content)

# Intermediate - 2
prompt = """
# Intermediate - 2
Solve the following logic puzzle step-by-step:
Four people (A, B, C, D) are sitting in a row. We know that:
1. A is not next to B.
2. B is next to C.
3. C is not next to D.

Determine the possible seating arrangements.

Step-by-step solution:
1. A 가 앉을 수 있는 가능한 자리를 모두 찾아
2. 1번 결과 중에서 B 가 앉을 수 있는 가능한 자리를 모두 찾아
3. 2번 결과 중에서 C 가 앉을 수 있는 가능한 자리를 모두 찾아
4. 3번 결과 중에서 D 가 앉을 수 있는 가능한 자리를 모두 찾아
5. 가능한 자리 조합 결과는 ABCD, BCDA 처럼 나열해
"""

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt,
        }
    ]
)

print(completion.choices[0].message.content)
