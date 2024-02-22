import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image

from lens import Lens, LensProcessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ii", "--input_image", type=str, required=True, help="Path to input image containing images")
    parser.add_argument("-ip", "--input_prompt", type=str, required=True, help="Prompt accompanying the image")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'")
    args = parser.parse_args()

    # Weights are loaded internally
    lens = Lens(device=args.device)
    lens.to(args.device)
    lens.eval()
    processor = LensProcessor()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side ='left', padding=True)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    llm_model.eval()
    llm_model.to(args.device)

    rgb_image = Image.open(args.input_image).convert('RGB')

    with torch.no_grad():
        samples = processor([rgb_image],[args.input_prompt])
        output = lens(samples)
        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
        if "cuda" in args.device:
            input_ids = input_ids.to(args.device)
        outputs = llm_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0])

    print(answer)
