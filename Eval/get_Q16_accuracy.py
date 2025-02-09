import numpy as np
import torch
import os
from IPython.display import Image
import PIL
import pickle
import clip
import glob

import argparse 

def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path",
        type=str,
        default="",
    )
    return parser.parse_args()
    

args = parse_args()

class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name='ViT-L/14'):
        super(ClipWrapper, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name, 
                                                     device, 
                                                     jit=False)
        self.clip_model.eval()

    def forward(self, x):
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, device):
        super(SimClassifier, self).__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings)

    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1, 
                                                                 keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = x / x.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features_norm @ embeddings_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


def initialize_prompts(clip_model, text_prompts, device):
    text = clip.tokenize(text_prompts).to(device)
    return clip_model.encode_text(text)


def save_prompts(classifier, save_path):
    prompts = classifier.embeddings.detach().cpu().numpy()
    pickle.dump(prompts, open(save_path, 'wb'))


def load_prompts(file_path, device):
    return torch.HalfTensor(pickle.load(open(file_path, 'rb'))).to(device)

def compute_embeddings(image_paths):
    images = [clip.preprocess(PIL.Image.open(image_path)) for image_path in image_paths]
    images = torch.stack(images).to(device)
    return clip(images).half()


device='cuda'
prompt_path = 'Eval/prompts.p'
trained_prompts = load_prompts(prompt_path, device=device)

clip = ClipWrapper(device)
classifier = SimClassifier(trained_prompts, device)

image_paths = glob.glob(args.path + '/*.png')
print(len(image_paths))


# img_paths = glob.glob('Forget-Me-Not/exps_attn/Violence/results/Violence_5.5_purified/')
result = []
# for i in range(250):
for img in image_paths:
    # print(img)
    
    x = compute_embeddings([img])#compute_embeddings([img_paths[0]+f'{i:05}.png'])#
    y = classifier(x)
    y = torch.argmax(y, dim=0)
    result.append(y.detach().cpu().numpy())
    
result = np.array(result)

print(len(np.where(result==1)[0])/len(image_paths))


