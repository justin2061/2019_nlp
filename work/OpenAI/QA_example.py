import os
import openai
from dotenv import load_dotenv

#-- ref url: https://beta.openai.com/examples/default-qa
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

content = '''
I am a highly intelligent question answering bot. 
If you ask me a question that is rooted in truth, I will give you the answer. 
If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".

Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: Unknown

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: How many squigs are in a bonk?
A: Unknown

Q: Where is the Valley of Kings?
A:
'''

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=content,
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

print(f'response: {response}')

# response: {
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "logprobs": null,
#       "text": "The Valley of Kings is located in Luxor, Egypt."
#     }
#   ],
#   "created": 1669973582,
#   "id": "cmpl-6Iwhi4TCGBeDuJHLJ7215gDVgF7fr",
#   "model": "text-davinci-003",
#   "object": "text_completion",
#   "usage": {
#     "completion_tokens": 12,
#     "prompt_tokens": 239,
#     "total_tokens": 251
#   }
# }