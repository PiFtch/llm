import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
############# code changes ###############
# import ipex
import intel_extension_for_pytorch as ipex
# verify Intel Arc GPU
print(ipex.xpu.get_device_name(0))
##########################################

# load model
model_id = "meta-llama/Llama-2-7b-hf"
dtype = torch.float16

# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
# tokenizer = LlamaTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained("C:\\Users\\Chenghong\\source\\repos\\llama-7b-hf", torch_dtype=dtype, low_cpu_mem_usage=True)
# tokenizer = LlamaTokenizer.from_pretrained("C:\\Users\\Chenghong\\source\\repos\\llama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("C:\\Users\\Chenghong\\source\\repos\\Llama-2-7b-hf", torch_dtype=dtype, low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained("C:\\Users\\Chenghong\\source\\repos\\Llama-2-7b-hf")
############# code changes ###############
# move to Intel Arc GPU
model = model.eval().to("xpu")
##########################################

# generate 
with torch.inference_mode(), torch.no_grad(), torch.autocast(
        ############# code changes ###############
        device_type="xpu",
        ##########################################
        enabled=True,
        dtype=dtype
    ):
    text = "You may have heard of Schrodinger cat mentioned in a thought experiment in quantum physics. Briefly, according to the Copenhagen interpretation of quantum mechanics, the cat in a sealed box is simultaneously alive and dead until we open the box and observe the cat. The macrostate of cat (either alive or dead) is determined at the moment we observe the cat."
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to("xpu")
    ##########################################
    generated_ids = model.generate(input_ids, max_new_tokens=128)[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_text)
