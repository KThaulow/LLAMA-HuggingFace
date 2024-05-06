import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)


# Chatting with the model
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "./trained_model",
                                                   device_map='auto',
                                                   quantization_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = getattr(torch, "float16"), bnb_4bit_quant_type = "nf4"))

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "./trained_model", trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

user_prompt = "Please tell me about Bursitis"
text_generation_pipeline = pipeline(task = "text-generation", model = model, tokenizer = tokenizer, max_length = 300)
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])