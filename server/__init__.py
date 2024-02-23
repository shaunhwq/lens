import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import json
from threading import Timer

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from lens import Lens, LensProcessor
from . import conversions


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_loaded = False
lens = None
processor = None
tokenizer = None
llm_model =  None
print(f"Using device: {device}")


def load_model():
    print("Loading models...")
    global lens, processor, tokenizer, llm_model, model_loaded, device
    lens = Lens()
    lens.to(device)
    lens.eval()
    processor = LensProcessor()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side ='left', padding=True)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    llm_model.eval()
    llm_model.to(device)
    model_loaded = True
    print("Models loaded.")


t = Timer(2.0, load_model)
t.start()

fast_api_app = FastAPI()
fast_api_app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@fast_api_app.get("/ping", response_class=JSONResponse)
async def ping():
    return {"message": "pong"}


@fast_api_app.post("/predict", response_class=JSONResponse)
async def predict(request: Request):
    if not model_loaded:
        return JSONResponse({"error": "model not loaded"}, status_code=500)

    # Retrieve image and question from body
    body = await request.body()
    json_obj = json.loads(body.decode("utf-8"))
    image_str = json_obj.get("image", None)
    question = json_obj.get("question", None)

    if image_str is None or question is None:
        return JSONResponse({"Make sure input has fields 'image' and 'question'"}, status_code=500)
    image = conversions.b64str_to_cv2_image(image_str)

    rgb_image = Image.fromarray(image).convert("RGB")

    with torch.no_grad():
        samples = processor([rgb_image],[question])
        output = lens(samples)
        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
        if "cuda" in device:
            input_ids = input_ids.to(device)
        outputs = llm_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0])

    return {"answer": answer}


@fast_api_app.get('/model-status')
async def get_model_status():
    return {"model_loaded": model_loaded}
