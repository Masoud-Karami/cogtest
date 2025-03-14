import os
import torch
import transformers
from huggingface_hub import HfApi, HfFolder, InferenceApi
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from ..base_classes import LLM 

class HF_API_LLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        engine, max_tokens, temperature = llm_info
        
        
        # Adapt engine name to Hugging Face API. Here I prepend hf_ to the engine name for all the hf models.
        if engine.startswith('hf_'):
            engine = engine.split("hf_")[1] # remove the "hf_" part
        padtokenId = 50256 # Falcon needs that to avoid some annoying warning
        
        # if engine.startswith("falcon"):
        #     engine = "tiiuae/" + engine 
        # elif engine.startswith("Yi"):
        #     engine = "01-ai/" + engine
        # elif (engine.startswith("mixtral")) or (engine.startswith("Mixtral")) or (engine.startswith("mistral")) or (engine.startswith("Mistral")):
        #     engine = "mistralai/" + engine
        # elif engine.startswith("mpt"):
        #     engine = "mosaicml/" + engine
        # elif engine.startswith("vicuna"):
        #     engine = "lmsys/" + engine
        # elif engine.startswith("koala"):
        #     engine = 'TheBloke/' + engine
        # elif ('longlora' in engine) or ('Alpaca' in engine):
        #     engine = 'Yukang/' + engine
        # elif 'CodeLlama' in engine:
        #     engine = 'codellama/' + engine + '-hf'
        # elif engine.startswith('llama-2'):
        
        hf_model_map = {
            "falcon": "tiiuae/",
            "Yi": "01-ai/",
            "mixtral": "mistralai/",
            "mistral": "mistralai/",
            "mpt": "mosaicml/",
            "vicuna": "lmsys/",
            "koala": "TheBloke/",
        }
            #Change llama-2-* to meta-llama/Llama-2-*b-hf
        for key, prefix in hf_model_map.items():
            if engine.startswith(key):
                engine = prefix + engine
                break
        if 'longlora' in engine or 'Alpaca' in engine:
            engine = 'Yukang/' + engine
        elif 'CodeLlama' in engine:
            engine = 'codellama/' + engine + '-hf'
        elif engine.startswith('meta-llama-2'):
            if 'chat' in engine:
                engine = 'meta-llama/L' + engine[1:].replace('-chat', '') + 'b-chat-hf'
            else:
                engine = 'meta-llama/L' + engine[1:] + 'b-hf'
        else:
            raise NotImplementedError(f"Unknown HF model: {engine}")

        engine=os.getenv('TRANSFORMERS_CACHE')+engine
        print(engine)
        
        print(f"Loading model: {engine}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(engine)
        except Exception as e:
            print(f"Tokenizer error: {e}")
            tokenizer = None  # Some models might not require tokenizers

        try:
            self.pipe = pipeline(
                "text-generation",
                model=engine,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                pad_token_id=padtokenId,
                max_new_tokens=max_tokens
            )
        except Exception as e:
            print(f"Pipeline initialization error: {e}")
            self.pipe = None  # Ensure pipeline isn't used if initialization fails

        if self.pipe:
            self.pipe.model.config.temperature = temperature + 1e-6

    def _generate(self, texts, temp, max_tokens):
        if not self.pipe:
            raise RuntimeError("Pipeline is not initialized.")

        # If `texts` is a single string, convert to list
        if isinstance(texts, str):
            texts = [texts]

        # Batch process multiple inputs instead of sequentially
        responses = self.pipe(texts, batch_size=len(texts))
        
        return [resp['generated_text'][len(text):] for text, resp in zip(texts, responses)]
