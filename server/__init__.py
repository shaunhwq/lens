import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from lens import Lens, LensProcessor

# Weights are loaded internally
print("Loading models...")
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
lens = Lens()
lens.to(device)
lens.eval()
processor = LensProcessor()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side ='left', padding=True)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
llm_model.eval()
llm_model.to(device)
print("Models loaded.")

# Set up FastAPI server
from .routes import router

fast_api_app = FastAPI()
fast_api_app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
fast_api_app.include_router(router)
