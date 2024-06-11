import vllm
import argparse
import os 
import json 
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


CODELLAMA_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

def code_llama_tokenizer_chat(
        message,
        chat_history=[]
    ):

    texts = [f'<s>[INST] <<SYS>>\n{CODELLAMA_SYSTEM_PROMPT}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    
    prompt = ''.join(texts)
    return prompt


def load_data( args):
    data_path = args.data_path 
    print(data_path)
    filenames = [x for x in os.listdir(data_path) if x.endswith('.jsonl')]
    print(filenames)
    list_data_dict = []
    
    for filename in filenames:
        file_path = os.path.join(data_path, filename)
        list_data_dict += utils.jload(file_path)


    if 'Phind' in args.base_model.lower():

        template = '''\
### System Prompt
You are an intelligent programming assistant.

### User Message
{instruction}

### Assistant Response
'''     
        print('+++++++++++++++++++Phind')
        prompts = [template.format(instruction=example["instruction"]) for example in list_data_dict]

    elif "codellama" in args.base_model.lower():
        if 'instruct' in args.base_model.lower():
            prompts = [code_llama_tokenizer_chat(example["instruction"]) for example in list_data_dict]
        else:
            # self.input_ids =[tokenizer(example["instruction"], return_tensors="pt")[0].squeeze() for example in list_data_dict] 
            print('base model:', args.base_model.lower())
            for example in list_data_dict:
                if len(example['prompt']) == 0:
                    print(example)
            prompts =[example["prompt"] for example in list_data_dict] 

    elif 'sft-codeqwen' in args.base_model.lower():
        template ='''\ 
<|im_start|>system
You are an AI programming assistant,  you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:<|im_end|>
<|im_start|>assistant
'''
        prompts = [template.format(instruction=example["instruction"]) for example in list_data_dict]

        print('sft-codeqwen')
        print(prompts[:2])

    elif 'finetune_dpsk' in args.base_model.lower():
        template ='''\
You are an AI programming assistant,  you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:
'''
        prompts = [template.format(instruction=example["instruction"]) for example in list_data_dict]

        print('sft-dpsk')
        print(prompts[:2])


    elif 'qwen' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example['instruction']}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print('qwen:', prompt)

            prompts.append(prompt)

    elif 'phi-3' in args.base_model.lower():
        prompts = [f'<|user|>\n{example["instruction"]}<|end|>\n<|assistant|>' for example in list_data_dict]
        
    elif 'codegemma' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        for example in list_data_dict:

            tmp = [{ "role": "user", "content": example["instruction"]}]
            prompt = tokenizer.apply_chat_template(tmp, tokenize=False, add_generation_prompt=True)
            print(prompt)

            prompts.append(prompt)
        
        # prompts = [ for example in list_data_dict] 
    
    elif 'deepseek' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:

            tmp = [{ "role": "user", "content": example["instruction"]}]
            prompt = tokenizer.apply_chat_template(tmp, tokenize=False, add_generation_prompt=True)
            # print(prompt)

            prompts.append(prompt)

    elif 'llama-3' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        
        for example in list_data_dict:

            tmp = [{ "role": "user", "content": example["instruction"]}]
            prompt = tokenizer.apply_chat_template(tmp, tokenize=False, add_generation_prompt=True)
            print(prompt)

            prompts.append(prompt)
    elif 'wizard' in args.base_model.lower():

        template = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
        prompts = [template.format(instruction=example["instruction"]) for example in list_data_dict]


    elif 'nxcode' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            messages = [
                {"role": "user", "content": example['instruction']}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print('nxcode:', prompt)

            prompts.append(prompt)

    elif 'opencode' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            messages = [
                {"role": "user", "content": example['instruction']}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print('opencode:', prompt)
            prompts.append(prompt)

    elif 'codestral' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            messages = [
                {"role": "user", "content": example['instruction']}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print('codestral:', prompt)
            prompts.append(prompt)

    assert len(prompts) == len(list_data_dict)
    return prompts, list_data_dict

def run(args):
    prompts, raw_datas = load_data(args)
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=0.95, max_tokens=args.max_length)

    print("model:", args.base_model)
    model = vllm.LLM(model=args.base_model, tensor_parallel_size=8, trust_remote_code=True)

    outputs = model.generate(prompts, sampling_params)

    assert len(outputs) == len(raw_datas)
    
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        raw_datas[idx]["raw_generation"] = [generated_text]


    save_path = os.path.join(args.outdir, args.base_model.split('/')[-1]+'_'+args.task+'.jsonl')

    with open(save_path, 'w') as f:
        for item in raw_datas:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--temperature", default=1.0, type=str, help="config path")
    parser.add_argument("--task", default="complete", type=str, help="config path")
    parser.add_argument("--outdir", default="outputs", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    parser.add_argument("--max_length", type=int, default=1024, help="beam size")

    args = parser.parse_args()
    run(args)
