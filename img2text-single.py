import torch
from torch.utils.data import DataLoader
from dataset import ImgCapDataset
from tqdm.auto import tqdm
import json
from transformers import AutoProcessor, Blip2ForConditionalGeneration





def main():
    result = dict()
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map=f"cuda:0")

    # model.cuda(opts.local_rank)
    
    dataset = ImgCapDataset('/workspace/img-caption/open-images/imgs/train_0')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, pin_memory=True)
    device = torch.device(f"cuda:0")

    try:
        for i, (img, fname) in enumerate(tqdm(dataloader)):
            prompt = ["A photography of "]*img.shape[0]
            inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
            
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            prompt = [f"{prompt[i]}{generated_text[i]}, where " for i in range(img.shape[0])]
            inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            


            prompt = [f"{prompt[i]}{generated_text[i]}. The vibe of this image is " for i in range(img.shape[0])]
            inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            


            prompt = [f"{prompt[i]}{generated_text[i]}. The saturation of this image is " for i in range(img.shape[0])]
            inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        


            prompt = [f"{prompt[i]}{generated_text[i]}. And the brightness of this image is " for i in range(img.shape[0])]
            inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            ret = [prompt[i] + generated_text[i] + "." for i in range(img.shape[0])]
            for i in range(img.shape[0]):
                result[fname[i]] = ret[i]

            if ((i/31)%10000 == 0):
                with open('result.json', 'w') as fp:
                    json.dump(result, fp)

    except Exception as e:
        print(e)
        with open('result.json', 'w') as fp:
            json.dump(result, fp)

    with open('result.json', 'w') as fp:
        json.dump(result, fp)



if __name__ == "__main__":
    main()
    

   