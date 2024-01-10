import os
import csv

import torch
from diffusers import AutoPipelineForText2Image


def load_prompts(path):
    if os.path.basename(path) == 'ViLG-300.csv':
        def csv_to_dict(file_path):
            result_dict = {}
            with open(file_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=',')
                for row in csv_reader:
                    prompt = row['\ufeffPrompt']
                    text = row['文本']
                    category = row['类别']
                    source = row['来源']
                    result_dict[prompt] = {'prompt': prompt, 'text': text, 'category': category, 'source': source}
            return result_dict
        data = csv_to_dict(path).keys()
    else:
        return NotImplementedError
    return data


def main(
    model_id="runwayml/stable-diffusion-v1-5",
    prompt_path="assets/ViLG-300.csv",
    save_path=None,
):
    if save_path is None:
        save_path = os.path.join('saved', model_id.replace('/', '_'))
        os.makedirs(save_path, exist_ok=True)

    prompts = load_prompts(prompt_path)
    pipeline = AutoPipelineForText2Image.from_pretrained(model_id)
    pipeline.to(device='cuda', dtype=torch.float16)
    for i, prompt in enumerate(prompts):
        print(f'{i}|{len(prompts)}: {prompt}')
        image = pipeline(prompt).images[0]
        image.save(os.path.join(save_path, f'{i}.jpg'))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
