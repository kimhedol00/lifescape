
## 프로젝트 소개
+ 가구 사장님이 오픈 마켓에서 가구를 판매하려면 스튜디오에서 가구 사진을 촬영해야 하지만, 돈과 시간이 많이 듭니다.
+ 이러한 문제를 해결하기 위해서, Generative AI 기반으로 인테리어 컷을 생성하는 것을 목표로 합니다.
+ 가구 도메인에 특화된 AI모델을 개발하기 위해서 학습데이터를 구축하고, 모델 학습을 진행했습니다.
+ 또한 모델을 이용해서 효과적인 인테리어 컷을 생성하기 위해 Extension을 적용했습니다.
+ 마지막으로 API Tool을 개발해서, 쉽게 인테리어 컷을 생성할 수 있도록 했습니다.

## 폴더 구조
	- demo  
	| - prompt  
	| - random_reference
	| - output
	| | - nukki
	| | - mask
	| | - img2img
	| | | - {yy-mm-dd}{hh-mm-ss}/0.png, 1.png
	| | - txt2img
	| - gpt_caption.py
	| - demo.py
	| - websockets_api_example3.py
	| - inference
	| - requirements.txt	

## Requirement 및 Install 방법
	# 이미지 생성용 서버 설치 및 실행
 	git clone https://github.com/comfyanonymous/ComfyUI
	
 	cd ComfyUI
	pip install -r requirements.txt
	pip install opencv-python

	## Custom Node 설치
	cd custom_nodes
	git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet
	git clone https://github.com/Acly/comfyui-inpaint-nodes
	git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
	git clone https://github.com/Fannovel16/comfyui_controlnet_aux

	# 모델 다운로드
	cd ../models
	mkdir inpaint
	cd inpaint
	https://huggingface.co/lllyasviel/fooocus_inpaint/tree/main
	여기서 fooocus_inpaint_head.pth, inpaint_v25.fooocus.patch, inpaint_v26.fooocus.patch
	다운로드 후 ./ComfyUI/models/inpaint에 저장 
	
 	# 실행
	cd ../..
	python main.py --listen [--novram / --lowvram / --normalvram / --highvram]


	# 이미지 생성 webui 설치 및 실행
	## git 레포지토리 clone
	git clone https://github.com/kimhedol00/lifescape
	cd lifescape
	
	## 필수 패키지 설치
	pip install -r requirements.txt
	
	## 실행
	python3 demo.py
 


## 모델 다운로드 방법
+ 거실/침실 모델 경로 : ./ComfyUI/models/checkpoints/
  + sdxl 베이스 모델 다운로드 : https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
  + sdxl refiner 모델 다운로드 : https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
  + 거실 모델 다운로드 : https://huggingface.co/hedol/livingroom
  + 침실 모델 다운로드 : https://huggingface.co/hedol/bedroom
+ 브랜드별 모델 경로 : ./ComfyUI/models/loras/
  + 봄소와 모델 다운로드 : https://huggingface.co/hedol/loras
  + 에싸 모델 다운로드 : https://huggingface.co/hedol/loras

## 1. 모델 학습 방법
+ Stable Diffusion XL Base 1.0 모델을 Fine-Tuning 시켜서 거실, 그리고 침실에 특화된 모델을 개발했습니다.
+ 학습된 모델이 특정 사이트의 특징을 잘 표현하도록 하기 위해, 학습된 모델을 Fine-Tuning 시켰습니다.

#### 학습 데이터 구축
+ 오늘의집과 같이 한국적인 분위기가 나는 고화질, 고품질 스튜디오 사진 거실 1021장, 침실 586장을 학습시켰습니다.
+ GPT-4o를 통해 이미지에서 caption을 추출하고, hyperparameter를 조절하며 학습을 진행했습니다.


## 2. 인테리어 컷 생성 Flow

![image_process](https://github.com/user-attachments/assets/df56a16a-0866-4456-9214-90570ebc9fc9)

+ python에서 Comfy UI GUI 를 이용해서 이미지를 생성합니다.
+ 가구 사진, 가구 누끼, prompt를 이용해서 가구와 어울리는 인테리어 컷을 생성합니다.


#### 가구 누끼 따기
+ transparent-background를 이용해서 가구의 누끼를 땁니다.
+ (가구 사진 -> 가구 누끼 딴 사진 Image)

#### prompt 생성하기
+ prompt를 생성하는 방법은 다음 두 가지가 존재합니다.

  1. 사용자가 직접 prompt를 작성하기
  2. 참고하는 인테리어 컷 이미지를 업로드하고, gpt-4o를 이용해서 prompt를 생성하기.

#### 인테리어 컷 생성 원리
+ 가구 이미지로부터 transparent_background를 사용하여 자동으로 누끼를 생성합니다.
+ 사용자로부터 요청사항(prompt)를 받거나 참고 이미지(reference image), 브랜드 템플릿을 입력받아 가구에 어울리는 배경 컷을 생성합니다.
+ 가구 사진과 가구 누끼를 이용해서, 가구가 존재하는 부분만 ControlNet의 Depth와 Canny를 이용합니다.
  + 가구의 깊이 정보와 윤곽선 정보를 이용하여, 인테리어 컷을 생성하면서 가구가 변형되는 정도를 낮춥니다.
  + 배경을 생성할 때 주변 가구 및 오브제를 생성할 때, 메인 가구의 정보를 이용해서 자연스럽게 생성하는 빈도를 높입니다.
  + 하지만 가구를 제외한 영역은 Depth및 Canny정보가 존재하지 않으므로, 자유로운 인테리어 컷을 생성할 수 있습니다.
+ 가구가 변형되는 정도를 낮추기 위해, inpaint_v25.fooocus.patch 모델을 이용해서 인테리어 컷을 생성합니다.
  + Stable Diffusion Inpaint를 이용하면 원래 존재하는 가구에 새롭게 가구를 생성해서 변형되는 경우가 잦습니다.
  + Fooocus Inpaint를 이용하면 이러한 가구를 변형시키는 빈도가 낮아집니다.
+ ControlNet, Fooocus Inpaint, prompt, 학습된 Model을 이용해서 인테리어 컷을 생성합니다.
+ 생성된 인테리어 컷을 Stable Diffusion XL refiner 1.0 모델을 이용해서, 질감을 더욱 강조합니다.

![img2img](https://github.com/user-attachments/assets/505486d1-8f5c-4e5f-9a8f-24895ecab739)

## 3. 남은 과제
+ 전처리단계에서 누끼를 딴 이미지를 수정하는 기능
+ 원본 가구(상품)의 변형 없이 배경을 생성
+ 색이나 위치에 대한 요청사항(prompt)를 인식할 수 있는 기능
+ inpaint나 upscale과 같은 후처리 기능
