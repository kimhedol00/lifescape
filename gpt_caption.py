import os
import base64
import requests
import random
import io
from PIL import Image

def _encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def _save_base64_image(base64_image, output_path):
    image_data = base64.b64decode(base64_image)
    with open(output_path, "wb") as output_file:
        output_file.write(image_data)
    print(f"Image saved to {output_path}")

def reference_get_caption(image_paths):
    api_key = "api_key"
    custom_prompt = (
        "You are an expert in interior design and are tasked with generating detailed captions for a dataset of interior images. "
        "The captions should be descriptive and precise, providing enough detail to help understand the overall design and aesthetic of the room.\n"
        "Generate a caption in this format: (function of the room), (interior style of the room), (color of the wallpaper),(texture of the wallpaper),(color of the flooring),(material of the flooring), (furniture1 information), (furniture2 information), .. ,(furniture n information),  I want you to mention descriptively about the color and what color is the object only for wallpaper and flooring. Also, I want you to mention descriptively about the material or texture of the wallpaper and flooring. When explaining texture of the wallpaper, start with 'with'. Also, when explaining material of the flooring, start with 'with' and end with 'pattern'. In the furniture information, I want you to exclude pillow, blanket information" +
        "\nExample: Living room, modern style, light beige color wallpaper, with vertical wooden panel, medium brown color flooring, with herringbone wood pattern, beige linen sectional sofa, square black wood coffee table, dark gray plush rug"
    )

    selected_image_path = random.choice(image_paths)[0]
    img = Image.open(selected_image_path)
    ref_image = img.copy()
    img.close()
    base64_image = _encode_image_to_base64(ref_image)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": custom_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if 'choices' in response_json and response_json['choices']:
            caption = response_json['choices'][0]['message']['content'].strip()
            return caption
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    return "Failed to get caption"

def process_image_list(image_list):
    caption = reference_get_caption(image_list)
    return caption