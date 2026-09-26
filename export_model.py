"""
Run this as a NEW CELL at the end of Project8_2_1 (after trainer.train()).
It saves the fine-tuned weights + the token-length limits the app needs,
and optionally pushes everything to the Hugging Face Hub.

The model is ~2.3 GB, which is too big for GitHub, so the Hub is where it lives.
"""
import json, os

SAVE_DIR = "./model"

trainer.save_model(SAVE_DIR)          # best checkpoint (load_best_model_at_end=True)
tokenizer.save_pretrained(SAVE_DIR)

with open(os.path.join(SAVE_DIR, "inference_config.json"), "w") as f:
    json.dump({
        "encoder_max": RECOMMENDED_ENCODER_MAX,
        "decoder_max": RECOMMENDED_DECODER_MAX,
    }, f, indent=2)

print("Saved to", SAVE_DIR, "| encoder_max =", RECOMMENDED_ENCODER_MAX,
      "| decoder_max =", RECOMMENDED_DECODER_MAX)

# ---- Optional: push to the Hugging Face Hub ----
# 1) pip install huggingface_hub ; 2) huggingface-cli login (use a WRITE token)
# from huggingface_hub import HfApi
# REPO = "<your-hf-username>/pegasus-agnews-headlines"
# api = HfApi()
# api.create_repo(REPO, exist_ok=True)
# api.upload_folder(folder_path=SAVE_DIR, repo_id=REPO)
