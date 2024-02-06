import os
import csv
from tqdm import tqdm
from PIL import Image
from clip_interrogator import Config, Interrogator

def generate_captions(obj, data_path, image_batch_size=128):
    assert obj.caption_model is not None, "No caption model loaded."
    obj._prepare_caption()
    # samples = [os.path.join(data_path, directory, file) for directory in os.listdir(data_path) for file in os.listdir(os.path.join(data_path,directory))]
    samples = [os.path.join(dirpath, name) for dirpath, _, files in os.walk(data_path) for name in files if name.endswith((".jpg", ".png", "bmp"))]
    captions = []
    for i in tqdm(range(0, len(samples), image_batch_size), desc="Generating captions"):
        batch = samples[i:i+image_batch_size]
        images = [Image.open(image_path).convert('RGB') for image_path in batch]    
        inputs = obj.caption_processor(images=images, return_tensors="pt").to(obj.device)
        if not obj.config.caption_model_name.startswith('git-'):
            inputs = inputs.to(obj.dtype)
        tokens = obj.caption_model.generate(**inputs, max_new_tokens=obj.config.caption_max_length)
        captions.extend(obj.caption_processor.batch_decode(tokens, skip_special_tokens=True))  
    return captions, samples

dataset_dir = './cifar10-64/test'
ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k", caption_model_name="blip2-flan-t5-xl"))
captions, samples = generate_captions(ci, dataset_dir)

if not os.path.exists(os.path.join(dataset_dir,'metadata.csv')):
    with open(os.path.join(dataset_dir,'metadata.csv'), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['file_name', 'caption'])

with open(os.path.join(dataset_dir,'metadata.csv'), 'a') as csv_file:
    csv_writer = csv.writer(csv_file)
    for path, caption in zip(samples, captions):
        path_split = path.split(os.sep)
        image_name = path_split[-1]
        directory_name = path_split[-2]
        baseimage_name = os.path.join(directory_name, image_name)
        csv_writer.writerow([baseimage_name, caption])