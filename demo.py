import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import os
import pytz
import requests
import base64
import io
from pydantic import BaseModel
from PIL import Image, ImageOps
import random
import pandas as pd
from gpt_caption import *
from nukki_edit import ImageEditor
from transparent_background import Remover
from websockets_api_example3 import *

webuiurl = "stable diffusion webui gradio public link here"


uploaded_images = []
nukki_path = ""

#봄소와 이미지
bomsowa_images = [
    (cv2.cvtColor(cv2.imread("./images/templates/bomsowa0.jpg"), cv2.COLOR_BGR2RGB), "봄소와(소파)"),
    (cv2.cvtColor(cv2.imread("./images/templates/bomsowa1.jpg"), cv2.COLOR_BGR2RGB), "봄소와(소파)"),
    (cv2.cvtColor(cv2.imread("./images/templates/bomsowa2.jpg"), cv2.COLOR_BGR2RGB), "봄소와(소파)")
]

# 에싸소파 이미지
essa_images = [
    (cv2.cvtColor(cv2.imread("./images/templates/essa0.jpg"), cv2.COLOR_BGR2RGB), "에싸(소파)"),
    (cv2.cvtColor(cv2.imread("./images/templates/essa1.jpg"), cv2.COLOR_BGR2RGB), "에싸(소파)"),
    (cv2.cvtColor(cv2.imread("./images/templates/essa2.jpg"), cv2.COLOR_BGR2RGB), "에싸(소파)")
]

def get_gallery_images(template_select):
    if template_select == "봄소와":
        return bomsowa_images
    elif template_select == "에싸":
        return essa_images
    else:
        return []
    
def toggle_accordion_visibility(template_select):
    if template_select == "선택안함":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def clear_nukki():
    return None

def clear_output():
    return gr.Gallery(value=[], show_label=False, columns=1, height="auto", allow_preview=True, preview=True, interactive=True)

def update_output_gallery():
    return gr.update(interactive=False, preview=True)

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def upload_images(new_images):
    global uploaded_images
    uploaded_images.extend(new_images)  # Add new images to the existing list
    return uploaded_images

def clear_images():
    global uploaded_images
    uploaded_images = []  # Clear the list of images
    return uploaded_images

def random_image():
    reference_dir = './random_reference'

    reference_images = [f for f in os.listdir(reference_dir) if os.path.isfile(os.path.join(reference_dir, f))]
    if not reference_images:
        raise RuntimeError("No images found in the random_reference directory")
    random_reference_image = random.choice(reference_images)
    random_reference_image_path = os.path.join(reference_dir, random_reference_image)
    reference_image = Image.open(random_reference_image_path)

    return [reference_image]

def goto_img2img():
    image_path = nukki_path
    if os.path.exists(image_path):
        print(f"Loading image from path: {image_path}")
        image = Image.open(image_path)
        return gr.update(selected=1), [image]
    else:
        print(f"Image path is invalid or does not exist: {image_path}")
        return gr.update(selected=1), None

def random_caption(model_select):
    file_path = "./prompt/sofa_caption_demo.csv"
    if model_select == "침실 학습 모델":
        file_path = './prompt/bed_caption_demo.csv'

    df = pd.read_csv(file_path)
    random_prompt = df.sample(n=1).iloc[0]['text']

    return random_prompt

def random_template(template_select):
    file_path = "./prompt/essa.csv"
    if template_select == "봄소와":
        file_path = './prompt/bomsowa.csv'

    df = pd.read_csv(file_path)
    random_prompt = df.sample(n=1).iloc[0]['text']

    return random_prompt

def nukki(init_image):
    if isinstance(init_image, np.ndarray):
        init_image = Image.fromarray(init_image).convert("RGBA")
    
    init_image_rgb = init_image.convert("RGB")
    remover = Remover() 
    masked_image = remover.process(init_image_rgb, type='rgba')

    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')

    nukki_dir = f"output/nukki/{date_str}"
    os.makedirs(nukki_dir, exist_ok=True)
    nukki_image_path = os.path.join(nukki_dir, f"{time_str}.png")
    masked_image.save(nukki_image_path)
    global nukki_path 
    nukki_path = nukki_image_path
    return nukki_image_path

def txt2img(user_prompt, batch_size, reference_gallery, model_select):
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')

    model = "sd_xl_base_1.0.safetensors"
    token = ""
    reference_prompt = ""
    seed = -1

    if model_select == "거실 학습 모델":
        model = "1021img_base_model-000080.safetensors"
        token = "thvk, "
    elif model_select == "침실 학습 모델":
        model = "0808_claeo_bedroom-000100.safetensors"
        token = "claeo, "

    if reference_gallery != None:
        reference_prompt = process_image_list(reference_gallery)

    prompt = token +  reference_prompt + user_prompt
    print("prompt : ", prompt)
    payload = {
        "prompt": prompt,
        "seed" : seed,
        "steps" : 60,
        "sampler_name" : "DPM++ 3M SDE",
        "scheduler" : "Automatic",
        "override_settings" : {
            "sd_model_checkpoint" : model
        },
        "batch_size": batch_size,
        "height" : 1024,
        "width" : 1024,
        "refiner_checkpoint": "sd_xl_refiner_1.0.safetensors",
        "refiner_switch_at": 0.8,
    }

    response = requests.post(url=f'{webuiurl}/sdapi/v1/txt2img', json=payload)
    if response.status_code != 200:
        raise RuntimeError(response.status_code, response.text)
    
    output_dir = f'./output/txt2img/{date_str}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    r = response.json()
    if 'images' not in r or not r['images']:
        raise RuntimeError("No images found in the response")

    file_paths = []
    for idx, img_data in enumerate(r['images']):
        file_path = os.path.join(output_dir, f'{time_str}_{idx}.png')
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(img_data))
        file_paths.append(file_path)
    return file_paths

def txt2img_select(evt : gr.SelectData):
    return evt.value['image']['path']

def img2img_demo(user_prompt, image, batch_size, reference_gallery, model_select, template_select):
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    size = 1024
    ref_images = []
    model = "sd_xl_base_1.0.safetensors"
    
    image_path = image[0][0] 
    init_image = Image.open(image_path).convert('RGB')  

    token = ""
    reference_prompt = ""
    template_prompt = ""
    seed = -1
    lora = ""

    if model_select == "거실 학습 모델":
        model = "1021img_base_model-000080.safetensors"
        token = "thvk, "
    elif model_select == "침실 학습 모델":
        model = "0809_claeo_bedroom-000150.safetensors"
        token = "claeo, "

    if reference_gallery != None:
        reference_prompt = process_image_list(reference_gallery)

    if template_select == "봄소와":
        lora = "bomsowa80.safetensors"
        template_prompt = random_template(template_select)
    elif template_select == "에싸소파":
        lora = "essa80.safetensors"
        template_prompt = random_template(template_select)

    prompt = token +  reference_prompt + template_prompt + user_prompt
    print("prompt : ", prompt)

    remover = Remover()
    masked_image = remover.process(init_image)

    masked_dir = f"output/mask/{date_str}"
    os.makedirs(masked_dir, exist_ok=True)
    masked_path = os.path.join(masked_dir, f"{time_str}.png")
    masked_image.save(masked_path)

    
    output_dir = f'./output/img2img/{date_str}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    result = getImage(
        fmodel = model,
        fpos_prompt = prompt,
        finputImage = init_image,
        fmaskImage = masked_image,
	fbatch = batch_size,
	floramodel = lora,
    )
    file_paths = []
    for idx, img in enumerate(result):
        if idx >= batch_size:
            continue
        file_path = os.path.join(output_dir, f'{time_str}_{idx}.png')
        img.save(file_path)
        file_paths.append(file_path)
        print(f"Saved image {idx} at {file_path}")

    print("All images saved.")
    print(file_paths)
    return file_paths

def goto_modify(image):
    return gr.update(selected=2)

def img2img_select(evt : gr.SelectData):
    return evt.value['image']['path']

if __name__ == "__main__":

    #### Theme ####
    theme = gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="red",
    )

    with gr.Blocks(theme=theme) as demo:
        with gr.Tabs() as tabs:

            with gr.TabItem("누끼따기", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        preprocess_input_request = gr.Textbox(value="가구 이미지를 넣어주세요!", interactive=False, show_label=False)
                        preprocess_input_image = gr.Image(show_label=False)
                        
                    with gr.Column(scale=2):
                        preprocess_output_request = gr.Textbox(value="누끼가 마음에 드신다면 \"완료\" 버튼을 눌러주세요!", show_label=False, interactive=False)
                        preprocess_output_image = gr.Image(label="후처리된 이미지", interactive=False, show_label=False)
                with gr.Row():
                    with gr.Column(scale=2):
                            preprocess_nukki_generation_button = gr.Button("누끼따기", variant='primary')
                    with gr.Column(scale=1):
                            preprocess_nukki_edit_button = gr.Button(value="수정(구현 예정)", variant='primary')
                    with gr.Column(scale=1):
                            preprocess_nukki_button = gr.Button(value="완료", variant='primary')

            with gr.TabItem("이미지 생성(Img2Img)", id=1):
                with gr.Row():
                    with gr.Column(scale=3):
                        img2img_request = gr.Textbox(label="상품 이미지를 넣어주세요!", value="제품 이미지와 요청 사항을 넣고 \"생성\" 버튼을 눌러주세요!", show_label=False, interactive=False)
                        img2img_image = gr.Gallery(show_label=False, columns=1, height="auto", allow_preview=True, preview=True)
                        
                        img2img_reference_text = gr.Textbox(label="Reference Image", interactive=False, value="참고할 이미지를 넣어주세요!")
                        img_reference_gallery = gr.Gallery(interactive=False, show_label=False, allow_preview=True, preview=True, height="auto", columns=1)
                        with gr.Row():
                            img_upload_btn = gr.UploadButton(label="Upload Images", file_types=["image"], file_count="multiple")
                            img_clear_btn = gr.Button("Clear Gallery")
                        
                        img_upload_btn.upload(upload_images, inputs=img_upload_btn, outputs=img_reference_gallery)
                        img_clear_btn.click(clear_images, outputs=img_reference_gallery)
                        
                        img2img_prompt = gr.Textbox(label="요청사항을 적어주세요!", placeholder="(예시) Bedroom, contemporary style, green vertical wood paneling wallpaper, white painted wood flooring, beige storage bed")
                        with gr.Accordion(label="세부 설정(선택사항)", open=False):
                            img_batch_size = gr.Slider(minimum=1, maximum=4, value=1, step=1, interactive=True, label="한번에 만들 이미지의 수 : 1 ~ 4 ( 기본 1 )")
                        
                        img2img_test_button = gr.Button(value="테스트용 랜덤 이미지", variant='primary')

                    with gr.Column(scale=1):
                        img_model_select = gr.Radio(["sdxl-base모델", "거실 학습 모델", "침실 학습 모델"], label="사용할 모델을 선택해주세요!", value="sdxl-base모델", min_width=50)
                        img_template_select = gr.Radio(["선택안함", "봄소와", "에싸"], label="원하는 브랜드가 있다면 선택해주세요!", value="선택안함", min_width=50)
                        
                        img_template_gallery_accordion = gr.Accordion(label="브랜드 예시 보기", open=False, visible=False)
                        with img_template_gallery_accordion:
                            img_template_gallery = gr.Gallery(interactive=False, show_label=False, height="auto", columns=1, allow_preview=False, show_fullscreen_button=True)
                        
                        img2img_start_button = gr.Button(value="생성", variant='primary')
                        
                    with gr.Column(scale=3):
                        img_input_request = gr.Textbox(value="생성된 결과물입니다!", show_label=False, interactive=False)
                        img2img_output = gr.Gallery(label = "", show_label=False, columns=1, height="auto", allow_preview=True, preview=True, interactive=False)
                
            with gr.TabItem("이미지 생성(Txt2Img)", id=2):
                with gr.Row():
                    with gr.Row():
                        with gr.Column(scale=1):
                            txt_reference_textbox = gr.Textbox(value="레퍼런스 이미지를 넣어주세요!", show_label=False, interactive=False)
                            txt_reference_gallery = gr.Gallery(interactive=False, show_label=False, allow_preview=False, height="auto", columns=1)
                            with gr.Row():
                                txt_upload_btn = gr.UploadButton(label="Upload Images", file_types=["image"], file_count="multiple")
                                txt_clear_btn = gr.Button("Clear Gallery")
                            
                            txt_upload_btn.upload(upload_images, inputs=txt_upload_btn, outputs=txt_reference_gallery)
                            txt_clear_btn.click(clear_images, outputs=txt_reference_gallery)
                            
                        with gr.Column(scale=1):    
                            txt2img_prompt = gr.Textbox(label="요청사항을 적어주세요!")
                            txt_model_select = gr.Radio(["sdxl-base모델", "거실 학습 모델", "침실 학습 모델"], label="사용할 모델을 선택해주세요!", value="sdxl-base모델")
                            txt_caption_btn = gr.Button(value="예시 prompt 생성", variant='primary')
                            with gr.Accordion(label="세부 설정(선택사항)", open=False):
                                batch_size = gr.Slider(minimum=1, maximum=4, value=1, step=1, interactive=True, label="한번에 만들 이미지의 수 : 1 ~ 4 ( 기본 1 )")

                        txt2img_start_button = gr.Button(value="생성", variant='primary')
  
                    with gr.Column(scale=1):
                        txt2img_output = gr.Gallery(show_label=False, columns=2, height="auto", allow_preview=True, preview=True)
                        txt_selected_image = gr.Image(interactive=False, show_label=True, visible=False)
                        
            preprocess_nukki_generation_button.click(fn=nukki, inputs = preprocess_input_image, outputs=preprocess_output_image)
            preprocess_nukki_generation_button.click(fn=clear_nukki, outputs=preprocess_output_image)
            preprocess_nukki_button.click(fn=goto_img2img, outputs=[tabs, img2img_image])
            
            txt2img_start_button.click(fn=txt2img, inputs=[txt2img_prompt, batch_size, txt_reference_gallery, txt_model_select], outputs=txt2img_output)
            txt2img_output.select(fn=txt2img_select, inputs=None, outputs=txt_selected_image)
            txt_caption_btn.click(fn=random_caption, inputs=txt_model_select, outputs=txt2img_prompt)

            img2img_start_button.click(fn=img2img_demo, inputs=[img2img_prompt, img2img_image, img_batch_size, img_reference_gallery, img_model_select, img_template_select], outputs=img2img_output)
            img2img_start_button.click(fn=clear_output, outputs=img2img_output)
            img2img_test_button.click(fn=random_image, inputs=None, outputs=[img_reference_gallery])
            img_template_select.change(fn=get_gallery_images, inputs=img_template_select, outputs=img_template_gallery)
            img_template_select.change(fn=toggle_accordion_visibility, inputs=img_template_select, outputs=img_template_gallery_accordion)
            img2img_output.change(fn=update_output_gallery, outputs=img2img_output)

    demo.launch(server_port=7861, share=True)
