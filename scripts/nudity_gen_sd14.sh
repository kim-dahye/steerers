# Define variables
CONCEPT_ERASURE="naked, nude, bare, exposed, Stripped, topless, male genitalia, penis, buttocks"
PROMPT="datasets/i2p.csv"
# PROMPT="datasets/mma-diffusion-nsfw-adv-prompts.csv"
# PROMPT="datasets/p4dn_16_prompt.csv"
# PROMPT="datasets/unlearndiff.csv"
# PROMPT="datasets/Ringabell_K_77.csv"
# PROMPT="datasets/nudity-ring-a-bell.csv"

OUTDIR="i2p_result/sd14_exp4_layer9"

# Run the Python script
python nudity_detect_sd14.py \
    --concept_erasure "$CONCEPT_ERASURE" \
    --start_iter 0 \
    --prompt "$PROMPT" \
    --strength -0.5 \
    --outdir "$OUTDIR" 

