import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration



processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="cuda:0")

# setup device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
# load sample image
image = Image.open("/workspace/img-caption/0a0ef2cdd2d27bc4.jpg").convert("RGB")
# display(raw_image.resize((596, 437)))

# prepare the image
# model = model.half()
# model.to(device)

# prompt = "Question: Can you describe in detail the different elements and their interactions in the image? Answer: "
prompt = "A photography of "
inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
prompt = f"{prompt}{generated_text}, where "

print(generated_text)


# 'singapore'