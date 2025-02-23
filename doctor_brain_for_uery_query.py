import os
import base64
from dotenv import load_dotenv
from groq import Groq

# Load API key from environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')


def encode_image(image_path):
    with open(image_path,'rb') as images_file:
        return base64.b64encode(images_file.read()).decode('utf-8')
    
    
# Function to analyze image with user query using Groq API
def analyze_image_with_user_query(user_query, encoded_image, model="llama-3.2-90b-vision-preview"):
    client = Groq(api_key=GROQ_API_KEY)  # Initialize client with API key
    
    # Constructing the messages for the LLM
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},  # User's query
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},  # Encoded image
            ],
        }
    ]

    # Sending the request to the multimodal model
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content    


# # Function to analyze image with a predefined system prompt (no user query)
# # Function to analyze an image using the predefined SYSTEM_PROMPT
# IMAGE_SYSTEM_PROMPT = "Examine this image and provide a precise medical assessment in 1-2 sentences. Keep the response professional, clear, and to the point."

# def analyze_image_with_prompt(encoded_image, model="llama-3.2-90b-vision-preview"):
#     client = Groq(api_key=GROQ_API_KEY)  # Initialize Groq API client

#     # Construct messages for LLM
#     messages = [
#         {"role": "system", "content": IMAGE_SYSTEM_PROMPT},  # Use global prompt
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},  # Encoded image
#             ],
#         }
#     ]

#     # Sending request to the multimodal model
#     chat_completion = client.chat.completions.create(
#         messages=messages,
#         model=model
#     )

#     return chat_completion.choices[0].message.content  # Return AI Doctor's response
