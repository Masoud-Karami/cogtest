import os
import torch
import transformers
from huggingface_hub import HfApi, HfFolder, InferenceApi
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from ..base_classes import LLM

# os.environ["HF_HOME"] = os.path.expanduser("~/scratch/huggingface/") # only for computecanada directory Beluga


class HF_API_LLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        engine, max_tokens, temperature = llm_info

        # Define base model directory
        hf_home = os.getenv("HF_HOME", "~/scratch/huggingface/")
        hf_home = os.path.expanduser(hf_home)

        # Adapt engine name to Hugging Face API. Here I prepend hf_ to the engine name for all the hf models.
        if engine.startswith('hf_'):
            engine = engine.split("hf_")[1]  # remove the "hf_" part
        padtokenId = 50256  # Falcon needs that to avoid some annoying warning

        hf_model_map = {
            "falcon": "tiiuae/",
            "Yi": "01-ai/",
            "mixtral": "mistralai/",
            "mistral": "mistralai/",
            "mpt": "mosaicml/",
            "vicuna": "lmsys/",
            "koala": "TheBloke/",
        }
        # Change llama-2-* to meta-llama/Llama-2-*b-hf
        for key, prefix in hf_model_map.items():
            if engine.startswith(key):
                engine = prefix + engine
                break
        if 'longlora' in engine or 'Alpaca' in engine:
            engine = 'Yukang/' + engine
        elif 'CodeLlama' in engine:
            engine = 'codellama/' + engine + '-hf'
        elif engine.startswith('llama-2'):
            if 'chat' in engine:
                engine = 'meta-llama/L' + \
                    engine[1:].replace('-chat', '') + 'b-chat-hf'
            else:
                engine = 'meta-llama/L' + engine[1:] + 'b-hf'
        else:
            raise NotImplementedError(f"Unknown HF model: {engine}")

        # Build absolute path
        engine_path = os.path.join(hf_home, engine)
        print(f"[HF_API_LLM] Loading model from: {engine_path}")

        print(f"Loading model: {engine}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(engine_path)
        except Exception as e:
            print(f"Tokenizer error: {e}")
            tokenizer = None  # Some models might not require tokenizers

        try:
            self.pipe = pipeline(
                "text-generation",
                model=engine_path,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                pad_token_id=padtokenId,
                max_new_tokens=max_tokens,
                batch_size=1
            )
        except Exception as e:
            print(f"Pipeline initialization error: {e}")
            self.pipe = None  # Ensure pipeline isn't used if initialization fails

        if self.pipe:
            self.pipe.model.config.temperature = temperature + 1e-6

    def _generate(self, texts, temp, max_tokens):
        if not self.pipe:
            raise RuntimeError("Pipeline is not initialized.")

        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]

        responses = self.pipe(texts, batch_size=len(texts))
        results = [resp[0]['generated_text']
                   [len(text):] for text, resp in zip(texts, responses)]

        return results[0] if is_single_input else results
