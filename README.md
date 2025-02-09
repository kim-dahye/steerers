# Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations

[Project Page](https://steerers.github.io/) [arXiv](https://arxiv.org/abs/your-paper-id)
Official code implementation of "Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations", arXiv 2025.

<img src="./assets/main.png" alt="Steerers" width="80%">


# Environment setup
```
git clone https://github.com/kim-dahye/steerers.git
conda env create -f steerers.yaml
conda activate steerers 
```

# 0. Extract intermediate diffusion features
```
python collect_features/collect_i2p_sd14.py  # For unsafe concepts, SD 1.4
python collect_features/collect_i2p_sdxl.py  # For unsafe concepts, SDXL
python collect_features/collect_i2p_flux.py  # For unsafe concepts, FLUX
```
# 1. Train k-SAE
```
bash scripts/train_sd14_i2p.sh  # For unsafe concepts, SD 1.4
bash scripts/train_flux_i2p.sh  # For unsafe concepts, FLUX
```
# 2. Generate images using prompt
```
bash scripts/nudity_gen_sd14.sh  # For nudity concept, SD 1.4
bash scripts/violence_gen_sd14.sh  # For violence concept, SD 1.4
```
# 3. Evaluate unsafe concept removal
To evaluate, first download the appropriate classifier for each category and place it inside the ```eval``` folder:
- Nudity: download the [NudeNet Detector](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx)
- Violence: download the [prompts.p](https://github.com/ml-research/Q16/blob/main/data/ViT-L-14/prompts.p) for the Q16 classifier
Then, run the following commands:
```
python Eval/compute_nudity_rate.py --root i2p_result/sd14_exp4_layer9  # For nudity concept
python get_Q16_accuracy.py --path violence_result/sd14_exp4_layer9
```
# play with jupyter notebook
style_transfer.ipynb

