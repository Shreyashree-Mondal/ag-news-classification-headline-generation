"""
Inference helpers for the fine-tuned PEGASUS headline generator
(trained with the Project12 notebook on the full AG News training split).
"""
import html
import json
import os
import re
import time

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Decoding strategies evaluated in the notebook
STRATEGIES = {
    "beam": {
        "label": "Beam search (6 beams)",
        "params": {"num_beams": 6, "length_penalty": 2.0,
                   "no_repeat_ngram_size": 3, "early_stopping": True},
    },
    "greedy": {"label": "Greedy", "params": {"do_sample": False, "num_beams": 1}},
    "temperature_0.77": {"label": "Sampling, temperature 0.77",
                         "params": {"do_sample": True, "temperature": 0.77, "num_beams": 1}},
    "top_k_50": {"label": "Top-k sampling (k=50)",
                 "params": {"do_sample": True, "top_k": 50, "num_beams": 1}},
    "top_p_0.9": {"label": "Nucleus sampling (p=0.9)",
                  "params": {"do_sample": True, "top_p": 0.9, "num_beams": 1}},
}


# ---------------------------------------------------------------------------
# Text cleaning: matches Project12 training, where the description was fed
# to the model unaltered. Only HTML entities/tags, links and control
# characters are cleaned up so pasted text looks like the training input.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def postprocess(raw: str) -> str:
    """PEGASUS-CNN emits '<n>' between sentences; a headline is the first one."""
    text = raw.replace("<n>", "\n")
    first = next((line.strip() for line in text.split("\n") if line.strip()), "")
    # AG News stores apostrophes as ' #39;', and the model sometimes copies that
    first = re.sub(r"\s?#?39;s\b", "'s", first)        # possessive
    first = re.sub(r"\s?#?39;(?=\s|$)", "'", first)    # closing quote
    first = re.sub(r"#?39;", "'", first)                # anything else
    first = re.sub(r"\s+", " ", first).rstrip(" .")
    return first


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class HeadlineGenerator:
    def __init__(self, model_id: str, encoder_max: int = 128, decoder_max: int = 64,
                 mock: bool = False):
        self.model_id = model_id
        self.encoder_max = encoder_max
        self.decoder_max = decoder_max
        self.mock = mock
        self.device = "cpu"

        if mock:
            return

        import torch
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_id)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @classmethod
    def from_env(cls):
        mock = os.getenv("MOCK_MODEL", "0") == "1"
        model_id = os.getenv("MODEL_ID", "./model")
        enc, dec = 128, 64

        # inference_config.json is written by export_model.py next to the weights
        cfg_path = os.path.join(model_id, "inference_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            enc = cfg.get("encoder_max", enc)
            dec = cfg.get("decoder_max", dec)

        enc = int(os.getenv("ENCODER_MAX", enc))
        dec = int(os.getenv("DECODER_MAX", dec))
        return cls(model_id, enc, dec, mock=mock)

    def generate(self, text: str, strategy: str = "beam", seed: int | None = None) -> dict:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'.")

        cleaned = clean_text(text)
        start = time.perf_counter()

        if self.mock:
            raw = cleaned.split(".")[0][:120]
        else:
            if seed is not None:
                self.torch.manual_seed(seed)
            inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True,
                                    max_length=self.encoder_max).to(self.device)
            with self.torch.no_grad():
                ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.decoder_max,
                    **STRATEGIES[strategy]["params"],
                )
            raw = self.tokenizer.decode(ids[0], skip_special_tokens=True)

        return {
            "headline": postprocess(raw),
            "raw_output": raw,
            "cleaned_input": cleaned,
            "strategy": strategy,
            "strategy_label": STRATEGIES[strategy]["label"],
            "latency_ms": round((time.perf_counter() - start) * 1000),
        }
