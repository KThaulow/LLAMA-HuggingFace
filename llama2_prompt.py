import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)


# Chatting with the model
llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "./trained_model",
                                                   device_map='mps',
                                                   quantization_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = getattr(torch, "float16"), bnb_4bit_quant_type = "nf4"))

llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "./trained_model", trust_remote_code = True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

user_prompt = "Please tell me about Bursitis"
text_generation_pipeline = pipeline(task = "text-generation", model = llama_model, tokenizer = llama_tokenizer, max_length = 300)
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])