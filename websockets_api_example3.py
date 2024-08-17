#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import random
from datetime import datetime

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

saveAddress = "temp file save path ( must be absolute path )" 

 
def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

prompt_text = """
{
  "9": {
    "inputs": {
      "text": "thvk, Living room, contemporary style, light coral color wallpaper, with smooth texture, pale beige color flooring, with chevron wood pattern, light blue fabric sectional sofa, modern floor lamp with an arc design, geometric pastel rug",
      "clip": [
        "80",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "10": {
    "inputs": {
      "text": "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, username, error, sketch, incorrect shadows, shadow, shadowed, shade, shadeds",
      "clip": [
        "80",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "14": {
    "inputs": {
      "samples": [
        "100",
        0
      ],
      "vae": [
        "94",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "19": {
    "inputs": {
      "ckpt_name": "1021img_base_model-000080.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "33": {
    "inputs": {
      "image": "화이트_2_원본.jpeg",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "35": {
    "inputs": {
      "head": "fooocus_inpaint_head.pth",
      "patch": "inpaint_v25.fooocus.patch"
    },
    "class_type": "INPAINT_LoadFooocusInpaint",
    "_meta": {
      "title": "Load Fooocus Inpaint"
    }
  },
  "36": {
    "inputs": {
      "model": [
        "19",
        0
      ],
      "patch": [
        "35",
        0
      ],
      "latent": [
        "44",
        2
      ]
    },
    "class_type": "INPAINT_ApplyFooocusInpaint",
    "_meta": {
      "title": "Apply Fooocus Inpaint"
    }
  },
  "44": {
    "inputs": {
      "positive": [
        "90",
        0
      ],
      "negative": [
        "90",
        1
      ],
      "vae": [
        "19",
        2
      ],
      "pixels": [
        "33",
        0
      ],
      "mask": [
        "68",
        0
      ]
    },
    "class_type": "INPAINT_VAEEncodeInpaintConditioning",
    "_meta": {
      "title": "VAE Encode & Inpaint Conditioning"
    }
  },
  "68": {
    "inputs": {
      "image": "화이트_2_누끼.png",
      "channel": "alpha",
      "upload": "image"
    },
    "class_type": "LoadImageMask",
    "_meta": {
      "title": "Load Image (as Mask)"
    }
  },
  "80": {
    "inputs": {
      "switch": "On",
      "lora_name": "essa80.safetensors",
      "strength_model": 0.5,
      "strength_clip": 0.5,
      "model": [
        "19",
        0
      ],
      "clip": [
        "19",
        1
      ]
    },
    "class_type": "CR Load LoRA",
    "_meta": {
      "title": "💊 CR Load LoRA"
    }
  },
  "82": {
    "inputs": {
      "mask": [
        "68",
        0
      ]
    },
    "class_type": "InvertMask",
    "_meta": {
      "title": "InvertMask"
    }
  },
  "85": {
    "inputs": {
      "expand": 1,
      "tapered_corners": true,
      "mask": [
        "82",
        0
      ]
    },
    "class_type": "GrowMask",
    "_meta": {
      "title": "GrowMask"
    }
  },
  "87": {
    "inputs": {
      "strength": 1,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "9",
        0
      ],
      "negative": [
        "10",
        0
      ],
      "control_net": [
        "89",
        0
      ],
      "image": [
        "33",
        0
      ],
      "mask_optional": [
        "85",
        0
      ]
    },
    "class_type": "ACN_AdvancedControlNetApply",
    "_meta": {
      "title": "Apply Advanced ControlNet 🛂🅐🅒🅝"
    }
  },
  "89": {
    "inputs": {
      "control_net_name": "sdxl_depth.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "90": {
    "inputs": {
      "strength": 1,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "87",
        0
      ],
      "negative": [
        "87",
        1
      ],
      "control_net": [
        "91",
        0
      ],
      "image": [
        "33",
        0
      ],
      "mask_optional": [
        "85",
        0
      ]
    },
    "class_type": "ACN_AdvancedControlNetApply",
    "_meta": {
      "title": "Apply Advanced ControlNet 🛂🅐🅒🅝"
    }
  },
  "91": {
    "inputs": {
      "control_net_name": "sdxl_canny.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "94": {
    "inputs": {
      "ckpt_name": "sd_xl_refiner_1.0.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "97": {
    "inputs": {
      "ascore": 6,
      "width": 1024,
      "height": 1024,
      "text": "",
      "clip": [
        "94",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXLRefiner",
    "_meta": {
      "title": "CLIPTextEncodeSDXLRefiner"
    }
  },
  "100": {
    "inputs": {
      "add_noise": "enable",
      "noise_seed": 456919324482827,
      "steps": 60,
      "cfg": 8,
      "sampler_name": "dpmpp_3m_sde",
      "scheduler": "karras",
      "start_at_step": 48,
      "end_at_step": 60,
      "return_with_leftover_noise": "disable",
      "model": [
        "94",
        0
      ],
      "positive": [
        "101",
        0
      ],
      "negative": [
        "97",
        0
      ],
      "latent_image": [
        "102",
        0
      ]
    },
    "class_type": "KSamplerAdvanced",
    "_meta": {
      "title": "KSampler (Advanced)"
    }
  },
  "101": {
    "inputs": {
      "ascore": 6,
      "width": 1024,
      "height": 1024,
      "text": "",
      "clip": [
        "94",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXLRefiner",
    "_meta": {
      "title": "CLIPTextEncodeSDXLRefiner"
    }
  },
  "102": {
    "inputs": {
      "add_noise": "enable",
      "noise_seed": 456919324482827,
      "steps": 60,
      "cfg": 8,
      "sampler_name": "dpmpp_3m_sde",
      "scheduler": "karras",
      "start_at_step": 0,
      "end_at_step": 48,
      "return_with_leftover_noise": "disable",
      "model": [
        "36",
        0
      ],
      "positive": [
        "44",
        0
      ],
      "negative": [
        "44",
        1
      ],
      "latent_image": [
        "104",
        0
      ]
    },
    "class_type": "KSamplerAdvanced",
    "_meta": {
      "title": "KSampler (Advanced)"
    }
  },
  "104": {
    "inputs": {
      "amount": 2,
      "samples": [
        "44",
        3
      ]
    },
    "class_type": "RepeatLatentBatch",
    "_meta": {
      "title": "Repeat Latent Batch"
    }
  },
  "109": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "14",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}
"""

prompt = json.loads(prompt_text)


def getImage(
  fmodel=None, 
  fpos_prompt="thvk, best quality", 
  fnegative_prompt="worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, username, error, sketch, incorrect shadows, shadow, shadowed, shade, shadeds", 
  finputImage=None, 
  fmaskImage=None, 
  floramodel="", 
  fbatch=1,
  fbaseSeed=-1, 
  frefinerSeed=-1):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]

    prompt["19"]["inputs"]["ckpt_name"] = fmodel

    prompt["9"]["inputs"]["text"] = fpos_prompt
    prompt["10"]["inputs"]["text"] = fnegative_prompt

    #inputImageAdd = saveAddress + formatted_time + ".png"
    #maskImageAdd = saveAddress + formatted_time + " mask.png"

    inputImageAdd = saveAddress + "temp_image.png"
    maskImageAdd = saveAddress + "temp_mask.png"
  
    finputImage.save(inputImageAdd)
    fmaskImage.save(maskImageAdd)

    prompt["33"]["inputs"]["image"] = inputImageAdd
    prompt["68"]["inputs"]["image"] = maskImageAdd

    if floramodel == "":
        prompt["80"]["inputs"]["switch"] = "On"
    else:
        prompt["80"]["inputs"]["switch"] = "Off"
        prompt["80"]["inputs"]["lora_name"] = floramodel
        prompt["80"]["inputs"]["model_weight"] = 0.8
        prompt["80"]["inputs"]["clip_weight"] = 0.8

    prompt["100"]["inputs"]["noise_seed"] = random.randint(1, 100000000) if fbaseSeed == -1 else fbaseSeed
    prompt["102"]["inputs"]["noise_seed"] = prompt["100"]["inputs"]["noise_seed"] if frefinerSeed == -1 else frefinerSeed

    print("base seed:" + str(prompt["100"]["inputs"]["noise_seed"]))
    print("refiner seed:" + str(prompt["102"]["inputs"]["noise_seed"]))

    prompt["104"]["inputs"]["amount"]=fbatch

    images = get_images(ws, prompt)

    ret_image = []

    for node_id in images:
        for image_data in images[node_id]:
            ret_image.append(Image.open(io.BytesIO(image_data)))
    
    return ret_image

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

"""
image = Image.open('a2.jpeg')
mask = Image.open('a1.png')

getI = getImage(
  fmodel="1021img_base_model-000080.safetensors",
  fpos_prompt="thvk, Living room, modern style, light taupe color wallpaper, with vertical ribbed texture, soft beige color flooring, with smooth wood pattern, round white wool rug, white sculptural side table, minimalist wall art, semi-circular wall shelf, soft light, diffuse light, open shade",
  finputImage=image,  
  fmaskImage=mask,
  fbaseSeed = 456919324482822,
  fbatch=2
  ) 


current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]

for idx, img in enumerate(getI):
    img.save(str(idx) + "f.png")
"""
