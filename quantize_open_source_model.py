import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from eight_bit_quantizer import W8A16Linear


def generate_output(input_text, model, tokenizer):

    encoded_tokens = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(
        input_ids=encoded_tokens["input_ids"], max_length=20
    )

    print(generated_tokens)

    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
        0
    ]

    return generated_text


def generate_output_with_hidden_states(input_text, model, tokenizer):
    encoded_tokens = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        generated_tokens = model(**encoded_tokens)

    for idx, k in enumerate(generated_tokens.hidden_states):
        print(idx, "---> ", k.shape)

    # generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
    #     0
    # ]

    # return generated_text


def replace_with_linear_quantization(
    model, target_module: nn.Module, module_name_to_exclude=[]
):
    for name, child in model.named_children():
        if name in module_name_to_exclude:
            continue
        if isinstance(child, nn.Linear):
            bias = True if child.bias is not None else False
            new_module = target_module(
                child.in_features, child.out_features, bias, child.weight.dtype
            )

            new_module.bias = child.bias
            new_module.quantize(child.weight)

            setattr(model, name, new_module)
        else:
            replace_with_linear_quantization(
                child, target_module, module_name_to_exclude
            )

    return model


def get_memory_footprint(model):
    memory_footprint = 0
    for n, p in model.named_parameters():
        params = p.numel()
        if p.dtype == torch.float32:
            params = params * 4
        elif p.dtype in [torch.bfloat16, torch.float16]:
            params = params * 2
        elif p.dtype in [torch.int8]:
            params = params * 1

        memory_footprint += params

    return memory_footprint / (1024**2)


input_text = "def hello_world():"
model_id = "Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
model.config.output_hidden_states = True
tokenizer = AutoTokenizer.from_pretrained(model_id)
torch.save({"model": model.state_dict()}, "model_before_quantization.pt")
print(model, model.get_memory_footprint())

start_time = time.time()
print(generate_output(input_text, model, tokenizer))
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# print(generate_output_with_hidden_states(input_text, model, tokenizer))

print("After Quantization:\n\n ")


model = replace_with_linear_quantization(
    model, target_module=W8A16Linear, module_name_to_exclude=["lm_head"]
)
torch.save({"model": model.state_dict()}, "model_after_quantization.pt")
print(model)

start_time = time.time()
print(generate_output(input_text, model, tokenizer))
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
