import json

import torch
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from PIL import Image

from . import conversions
from . import device, lens, processor, tokenizer, llm_model

router = APIRouter()


@router.get("/ping", response_class=JSONResponse)
async def get_jobs():
    return {"message": "pong"}


@router.post("/predict", response_class=JSONResponse)
async def predict(request: Request):
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
