import os
import sys
import h5py
import copy
import json
import argparse
import logging
import re
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, Sequence, List
import jsonlines
import utils

import torch
import transformers
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)

from transformers import GenerationConfig
logging.basicConfig(level=logging.DEBUG)  
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


CODELLAMA_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEBUG=False





def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForInference(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 4096)
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def padding(inputs, padding_token, cutoff = None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    else:
        cutoff = min(cutoff, max([len(item) for item in inputs]))
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens


def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s) * pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)
    return gathered_s

def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    # print(tokenizer.model_max_length)
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))


def code_llama_tokenizer(
        dialogs,
        tokenizer
    ):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
    UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:
        unsafe_requests.append(
            any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
        )
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = False
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(torch.tensor(dialog_tokens))
    return prompt_tokens

def code_llama_tokenizer_chat(
        message,
        tokenizer,
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
    prompt_tokens = tokenizer([prompt], return_tensors='pt', add_special_tokens=False)
    # print(prompt_tokens)
    return prompt_tokens['input_ids']

def octocoder_tokenizer_chat(
        message,
        tokenizer,
        chat_history=[]
    ):

    prompt = f'Question:{message}\n\nAnswer:'
    prompt_tokens = tokenizer([prompt], return_tensors='pt', add_special_tokens=False)
    # print(prompt_tokens)
    return prompt_tokens['input_ids']


def build_codeshell_chat_input(query, history=[], tokenizer=None, max_new_tokens=None):
    user_name = "## human:"
    ai_name = "## assistant: "
    stop = '|<end>|'

    prompt = ''
    for q, r in history:
        prompt += f"{user_name}{q}{stop}"
        prompt += f"{ai_name}{r}{stop}"
    prompt += f"{user_name}{query}{stop}"
    prompt += ai_name.rstrip()


    max_new_tokens = max_new_tokens or 128
    # max_input_tokens = self.config.n_positions - max_new_tokens

    prompt_tokens = tokenizer([prompt], return_tensors='pt', add_special_tokens=False)
    # print(prompt_tokens)
    return prompt_tokens['input_ids']



def wizard_tokenizer_chat(
        message,
        tokenizer,
        chat_history=[]
    ):

    prompt = f'"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{message}\n\n### Response:"'
    prompt_tokens = tokenizer([prompt], return_tensors='pt', add_special_tokens=False)
    # print(prompt_tokens)
    return prompt_tokens['input_ids']

def build_humaneval_instruction(items):
    # f = open('tmp.json', 'w')
    instruct_template = """
Please continue to complete the function and return all completed code in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip()

    for item in items:
        lang = item['task_id'].split('/')[0].lower()
        if lang in ['awk', 'markdown', 'json', 'html']:
            item['humaneval_instruction'] = item['instruction']
            continue 
        if lang=='visual basic':
            lang = 'vb'
        elif lang=='shell':
            lang = 'bash'
        elif lang=='f#':
            lang = 'fsharp'
        elif lang =='emacs lisp':
            lang = 'emacs'
        elif lang =='c#':
            lang = 'csharp'
        elif lang == 'common lisp':
            lang = 'lisp'

        # else:
        #     print(lang)
        item['humaneval_instruction'] = instruct_template.format(lang.lower(), item['prompt'].strip())

        
    #     f.write(item['humaneval_instruction']+'\n\n')
    # f.close()


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, args):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        filenames = [x for x in os.listdir(data_path) if x.endswith('.jsonl')]
        print(filenames)
        list_data_dict = []
        
        for filename in filenames:
            file_path = os.path.join(data_path, filename)
            list_data_dict += utils.jload(file_path)

        # build_humaneval_instruction(list_data_dict)

        # list_data_dict = list_data_dict[:100]

        if list_data_dict[0].get("input_ids") is None:
            logging.warning("Tokenizing inputs... This may take some time...")
            if "deepseek" in args.base_model.lower():
                # print(tokenizer.apply_chat_template([{'role': 'user', 'content': list_data_dict[0]["instruction"]}]))
                if 'instruct' in args.base_model.lower():
                    self.input_ids = [tokenizer.apply_chat_template([{'role': 'user', 'content': example["instruction"]}], add_generation_prompt=True, return_tensors="pt")[0].squeeze() for example in list_data_dict]
                else:
                    # print('base model:', args.base_model.lower())
                    for example in list_data_dict:
                        if len(example['prompt']) == 0:
                            print(example)
                    self.input_ids =[tokenizer([example["prompt"]], return_tensors="pt")['input_ids'][0].squeeze() for example in list_data_dict] 
            elif 'codegemma' in args.base_model.lower():
                self.input_ids = [tokenizer.apply_chat_template([{'role': 'user', 'content': example["instruction"]}], add_generation_prompt=True, return_tensors="pt")[0].squeeze() for example in list_data_dict]
            elif "codellama" in args.base_model.lower():
                if 'instruct' in args.base_model.lower():
                    self.input_ids = [code_llama_tokenizer_chat(example["instruction"], tokenizer=tokenizer)[0].squeeze() for example in list_data_dict]
                else:
                    # self.input_ids =[tokenizer(example["instruction"], return_tensors="pt")[0].squeeze() for example in list_data_dict] 
                    print('base model:', args.base_model.lower())
                    for example in list_data_dict:
                        if len(example['prompt']) == 0:
                            print(example)
                    self.input_ids =[tokenizer([example["prompt"]], return_tensors="pt")['input_ids'][0].squeeze() for example in list_data_dict] 
            
            elif "magicoder" in args.base_model.lower():
                self.input_ids = [tokenizer.apply_chat_template([{'role': 'user', 'content': example["instruction"]}], add_generation_prompt=True, return_tensors="pt")[0].squeeze() for example in list_data_dict]
            # elif 
            elif "octocoder" in args.base_model.lower():
                self.input_ids = [octocoder_tokenizer_chat(example["instruction"], tokenizer=tokenizer)[0].squeeze() for example in list_data_dict]
            elif 'wizardcoder'in args.base_model.lower():
                self.input_ids = [wizard_tokenizer_chat(example["instruction"], tokenizer=tokenizer)[0].squeeze() for example in list_data_dict]
            elif  "codeqwen" in args.base_model.lower():
                if 'chat' in args.base_model.lower():
                    # print(args.base_model.lower())
                    
                    self.input_ids = []
                    for example in list_data_dict:
                        
                        text_tmp = tokenizer.apply_chat_template(
                        [   {"role": "system", "content": "You are a helpful assistant."},
                            {'role': 'user', 'content': example['instruction']}
                            ], add_generation_prompt=True, tokenize=False)
                        self.input_ids.append(tokenizer([text_tmp], return_tensors="pt")['input_ids'][0].squeeze() )
                else:
                    print('qwen base model')
                    for example in list_data_dict:
                        if len(example['prompt']) == 0:
                            print(example)
                    self.input_ids =[tokenizer([example["prompt"]], return_tensors="pt")['input_ids'][0].squeeze() for example in list_data_dict] 
            elif 'starcoder' in args.base_model.lower():
                if 'instruct' in args.base_model.lower():
                    self.input_ids = [tokenizer.apply_chat_template([{'role': 'user', 'content': example["instruction"]}], add_generation_prompt=True, return_tensors="pt")[0].squeeze() for example in list_data_dict]
                else:
                    for example in list_data_dict:
                        if len(example['prompt']) == 0:
                            print(example)
                    self.input_ids =[tokenizer([example["prompt"]], return_tensors="pt")['input_ids'][0].squeeze() for example in list_data_dict] 
            elif 'qwen' in args.base_model.lower():
                self.input_ids = []
                for example in list_data_dict:
                    text_tmp = tokenizer.apply_chat_template(
                    [   {"role": "system", "content": "You are a helpful assistant."},
                        {'role': 'user', 'content':example["instruction"]}
                        ], add_generation_prompt=True, tokenize=False)
                    # print(text_tmp)
                    self.input_ids.append(tokenizer([text_tmp], return_tensors="pt")['input_ids'][0].squeeze() )
                    del text_tmp 
            elif 'yi' in args.base_model.lower():
                print('yiyiyiyi')
                self.input_ids = [tokenizer.apply_chat_template([{'role': 'user', 'content': example["instruction"]}], return_tensors="pt")[0].squeeze() for example in list_data_dict]
            elif 'codeshell' in args.base_model.lower():
                print('codeshell')
                self.input_ids = [build_codeshell_chat_input(example["instruction"], tokenizer=tokenizer)[0].squeeze() for example in list_data_dict]
            elif 'codegen' in args.base_model.lower():
                print('codegen')


            else:
                print('unrecognized model')
            self.raw = list_data_dict
        else:
            logging.info("Loading tokenized sentences...")
            def truncate(sentence):
                return torch.tensor(sentence[:args.model_max_length] + [tokenizer.eos_token_id] if len(sentence) > args.model_max_length else sentence)
            self.input_ids = [truncate(example["test_input_ids"]) for example in list_data_dict]
            self.raw = list_data_dict
 

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i])


@torch.no_grad()
def main(rank, args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size


    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, args=args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token='[PAD]'),
            tokenizer=tokenizer,
            model=model,
        )

    tokenizer.truncation_side = 'right'
    torch.cuda.set_device(LOCAL_RANK)
    model.to(LOCAL_RANK)
    model = DDP(model, device_ids=[LOCAL_RANK])
    model.eval()

    data_collator = DataCollatorForInference(tokenizer=tokenizer)
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=WORLD_SIZE, rank=WORLD_RANK, shuffle=False)
    dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=data_collator, 
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
    )

        
    generation_config = GenerationConfig(
        temperature= 1.0,
        do_sample = False,
        top_p = 1.0,
        repetition_penalty = 1.0,
        max_new_tokens=args.maxlen_out,
        num_return_sequences=args.return_seq_num,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_ids = tokenizer.eos_token_id
    )
    all_outputs = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        generation_output = model.module.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        s = generation_output.sequences
        #s = set_empty_token(s, tokenizer.eos_token_id, tokenizer.pad_token_id)

        bsz = input_ids.shape[0]
        gather_outputs  = sequence_gather(s, WORLD_SIZE, tokenizer.pad_token_id)
        gathered_inputs = sequence_gather(input_ids, WORLD_SIZE, tokenizer.pad_token_id)
       
        gather_outputs  = torch.stack(gather_outputs).reshape(WORLD_SIZE, bsz, args.return_seq_num, -1)
        gathered_inputs = torch.stack(gathered_inputs)
        gather_outputs  = gather_outputs.transpose(0,1).reshape(bsz * WORLD_SIZE * args.return_seq_num, -1)

        gathered_inputs = gathered_inputs.transpose(0,1).reshape(bsz * WORLD_SIZE, -1)
        outputs_string  = tokenizer.batch_decode(gather_outputs , skip_special_tokens=True)
        inputs_string   = tokenizer.batch_decode(gathered_inputs, skip_special_tokens=True)


        for idx in range(len(inputs_string)):
            temp = []
            for i in range(args.return_seq_num):
                temp.append([inputs_string[idx], outputs_string[args.return_seq_num * idx + i].replace(inputs_string[idx], '')])        
            # print(temp[0])
            all_outputs.append(temp)
    
    all_outputs = all_outputs[:len(eval_dataset)]
    os.makedirs(args.out_path, exist_ok=True)
    if rank == 0:
        # assert len(all_outputs) == len(eval_dataset.raw)
        output_path = os.path.join(args.out_path, data_path.split('/')[-1])
        with jsonlines.open(output_path+'.jsonl', 'w') as w:
            for idx, (item, raw) in enumerate(zip(all_outputs, eval_dataset.raw)):

                raw["raw_generation"] = [item[0][-1].rstrip()]
                w.write(raw)

        print(f"Successfully saving to {output_path}")
    dist.barrier()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--chatml-format", default="code-llama", type=str, help="model path")
    parser.add_argument("--model_name", default="deepseek-coder-7B", type=str, help="model path")
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--temperature", default=1.0, type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--beam_size", type=int, default=1, help="beam size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--use_diverse_beam", type=bool, default=False, help="batch size")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    parser.add_argument("--model_type", default="cmt", type=str, help="config path")
    parser.add_argument("--model_max_length", type=int, default=8192, help="beam size")
    parser.add_argument("--dynamic_load_image", default=True, type=bool, help="config path")
    parser.add_argument("--return_seq_num", default=1, type=int, help="config path")
    parser.add_argument("--maxlen_out", default=1024, type=int, help="config path")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)