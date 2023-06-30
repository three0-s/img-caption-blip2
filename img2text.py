import torch
from PIL import Image
from torch.utils.data import DataLoader
from dataset import ImgCapDataset
from tqdm.auto import tqdm
import json
from transformers import AutoProcessor, Blip2ForConditionalGeneration



processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="cuda:0")


# setup device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

dataset = ImgCapDataset('/workspace/img-caption/open-images/imgs/train_0')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False)



result = dict()
for i, (img, fname) in enumerate(tqdm(dataloader)):
    prompt = "A photography of "
    inputs = processor(img.squeeze(0), text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    prompt = f"{prompt}{generated_text}, where "
    inputs = processor(img.squeeze(0), text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    


    prompt = f"{prompt}{generated_text}. The vibe of this image is "
    inputs = processor(img.squeeze(0), text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    


    prompt = f"{prompt}{generated_text}. The saturation of this image is " 
    inputs = processor(img.squeeze(0), text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
   


    prompt = f"{prompt}{generated_text}. And the brightness of this image is "
    inputs = processor(img.squeeze(0), text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    ret = prompt + generated_text + "."

    result[fname[0]] = ret
    if (i%10000 == 0):
        with open('result.json', 'w') as fp:
            json.dump(result, fp)
