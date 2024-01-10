import gradio as gr
import os
import csv

root = 'saved'
prompt_path = 'assets/ViLG-300.csv'


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
        data = list(csv_to_dict(path).keys())
    else:
        return NotImplementedError
    return data


prompts = load_prompts(prompt_path)


def load_images(methods, idx):
    idx = int(idx)
    prompt = prompts[idx].strip()
    images = []
    for method in methods:
        image = os.path.join(root, method, f'{idx}.jpg')
        images.append((image, method))
    return prompt, images


def load_methods():
    methods = os.listdir(root)
    return methods


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Text to Image Models Comparison")
        with gr.Row():
            idx = gr.Number(value=0, label='Index')
            prompt = gr.Textbox()
        methods = gr.Dropdown(multiselect=True, choices=load_methods(), value=load_methods(), label='Methods')
        gallery = gr.Gallery(show_label=False, object_fit='fill', height=600, columns=5)
        idx.change(load_images, [methods, idx], [prompt, gallery])
        methods.change(load_images, [methods, idx], [prompt, gallery])
    demo.launch()


if __name__ == '__main__':
    main()
