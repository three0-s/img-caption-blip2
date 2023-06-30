import torch
import argparse
from torch.utils.data import DataLoader
from dataset import ImgCapDataset
from tqdm.auto import tqdm
import json
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--local-rank', dest='local_rank', type=int)
    # usage : --gpu_ids 0, 1, 2, 3
    return parser


def main(opts):
    torch.cuda.set_device(opts.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    result = dict()
    opts.world_size = torch.distributed.get_world_size()
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map=f"cuda:{opts.local_rank}")

    # model.cuda(opts.local_rank)
    model = DDP(model, delay_allreduce=True).cuda(opts.local_rank)
    dataset = ImgCapDataset('/workspace/img-caption-blip2/open-images/')
    sampler = DistributedSampler(dataset=dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=False, sampler=sampler)
    device = torch.device(f"cuda:{opts.local_rank}")

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
    except:
        with open('result.json', 'w') as fp:
            json.dump(result, fp)

    with open('result.json', 'w') as fp:
            json.dump(result, fp)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('BLIP2 distributed inference for image-to-text', parents=[get_args_parser()])
    opts = parser.parse_args()
    main(opts)
    

   