import torch


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, OPTForCausalLM

model_id = "facebook/opt-125m"


config = AutoConfig.from_pretrained(model_id)

with torch.device("meta"):
    model = OPTForCausalLM(config)


print(model)
