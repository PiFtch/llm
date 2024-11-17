import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

############# code changes ###############
# import ipex
import intel_extension_for_pytorch as ipex
# verify Intel Arc GPU
print(ipex.xpu.get_device_name(0))
##########################################

# load model
model_id = "meta-llama/Llama-2-7b-chat-hf"
dtype = torch.float16

# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
# tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained("C:\\Users\\Chenghong\\source\\repos\\Llama-2-7b-chat-hf", torch_dtype=dtype, low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained("C:\\Users\\Chenghong\\source\\repos\\Llama-2-7b-chat-hf")

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
    text = "Humans have good generalization abilities. For example, children who have learned how to calculate 1+2 and 3+5 can later calculate 15 + 23 and 128 x 256. Can deep learning have such generalization ability like humans do?"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to("xpu")
    ##########################################
    generated_ids = model.generate(input_ids, max_new_tokens=512)[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_text)
