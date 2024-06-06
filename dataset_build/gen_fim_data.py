import os 
import json 
import random 
import copy 
random.seed(10)

PROMPT_TEMPLATE='''\
Below is a explanation of {lang} code and incomplete code implementation.

* Docstring: 
{docstring}

* Incomplete Code:
{code}

Please fill the [MASK]（multiple lines of code may be masked out) and write the complete function.'''

def is_valid(line:str, lang=None):
    line = line.strip()
    stop_words = ['}', '{', 'end', 'else','endif', 'if']
    if line in stop_words:
        return False 
    if len(line) <4:
        return False
    if lang in ['JSON', 'Markdown']: 
        return True 
    if (line.startswith('*') or 
        line.startswith('#') or 
        line.startswith('!') or 
        line.startswith('\\') or 
        line.startswith('\'') or 
        line.startswith('\"') or 
        line.startswith('~') or 
        line.startswith('@') or 
        line.startswith('end') or 
        line.startswith('stop') or 
        line.startswith('/')):
        return False  
    return True 

def gen_lines_fim_data(item, mask_line_cnt=1):
    # signature = item['signature']
    lang = item['task_id'].split('/')[0]
    solution = item['canonical_solution']
    # docstring = item['docstring']

    code_lines = solution.split('\n') 
    if len(code_lines)<=3:
        return None
    
    valid_line_no = []
    for idx, line in enumerate(code_lines):
        if is_valid(line, lang=lang):
            valid_line_no.append(idx)
    if len(valid_line_no) <=3:
        return None 
    mask_no = random.sample(valid_line_no, k=min(mask_line_cnt, len(valid_line_no)-2))
    
    for no in mask_no:
        code_lines[no] = '[MASK]'
    
    masked_code = '\n'.join(code_lines)
    return masked_code


def gen_span_fim_data(item):
    # signature = item['signature']
    lang = item['task_id'].split('/')[0]
    solution = item['canonical_solution']
    # docstring = item['docstring']

    code_lines = solution.split('\n') 

    valid_line_no = []
    for idx, line in enumerate(code_lines):
        if is_valid(line,lang=lang):
            valid_line_no.append(idx)

    if not(valid_line_no):
        print(item)
        return None 
    if len(valid_line_no) <=3 :
        mask_line_idx = random.sample(valid_line_no, k=1)[0]
        mask_line = code_lines[mask_line_idx]
        split_start = random.randint(1, (len(mask_line)-1)//2)
        split_end = split_start+int(len(mask_line)*0.4)
        mask_line = mask_line[:split_start]+'[MASK]'+mask_line[split_end:]
        code_lines[mask_line_idx] = mask_line 
        return '\n'.join(code_lines)
    
    mask_sta_line_idx, mask_end_line_idx = sorted(random.sample(valid_line_no, k=2))
    
    sta_line = code_lines[mask_sta_line_idx] 
    end_line = code_lines[mask_end_line_idx] 
    
    split_sta = random.randint(0, len(sta_line)-1)
    split_end = random.randint(0, len(end_line)-1)

    if mask_end_line_idx == len(code_lines)-1:
        code_lines = code_lines[:mask_sta_line_idx]+[sta_line[:split_sta]+'[MASK]'+end_line[split_end:]]
    else:
        code_lines = code_lines[:mask_sta_line_idx]+[sta_line[:split_sta]+'[MASK]'+end_line[split_end:]]+ code_lines[mask_end_line_idx+1:]

    masked_code = '\n'.join(code_lines)
    return masked_code



if __name__ == '__main__':

    file_fix = {'HTML':'html','Markdown':'md','JSON':'json'}
    # single_save_dir = './fim_data/single'
    # multi_save_dir  = './fim_data/multi'
    # span_save_dir   = './fim_data/span'
    # merge_save_dir   = './fim_data/merge'
    # files = os.listdir('./data')
    single_save_dir  = './fim_data_part2/single'
    multi_save_dir   = './fim_data_part2/multi'
    span_save_dir    = './fim_data_part2/span'
    merge_save_dir   = './fim_data_part2/merge'

    light_span_save_dir = './fim_data_part2/light'

    # data_dir = '/workspace/MultiCodeEval/code/fim_data_add'
    data_dir = '/workspace/MultiCodeEval/code/MMCodeEval/data'
    files = os.listdir(data_dir)
    files = [x for x in files if x.endswith('.jsonl')]

    file_cnt = 0

    GEN_COUNT=2
    for file in files:     
        file_cnt+=1
        file_path = os.path.join(data_dir, file)
        items = [json.loads(x) for x in open(file_path).readlines() if x]
        print(file, len(items))  
        single_save_path = os.path.join(single_save_dir, file)
        multi_save_path  = os.path.join(multi_save_dir, file)
        span_save_path   = os.path.join(span_save_dir, file)
        merge_save_path   = os.path.join(merge_save_dir, file)
        light_span_save_path   = os.path.join(light_span_save_dir, file)
        f_single =  open(single_save_path,'w')
        f_multi  =  open(multi_save_path,'w')
        f_span   =  open(span_save_path,'w')
        f_merge   =  open(merge_save_path,'w')
        f_light_span = open(light_span_save_path, 'w')

        single_line_items  = []
        multi_line_items   = []
        rand_span_items    = []
        light_span_items   = []

        ############# mask single line ###################
        for item in items:
            lang = item['task_id'].split('/')[0]
            # if 'signature' not in item or 'canonical_solution' not in item or 'docstring' not in item:
            if 'canonical_solution' not in item: # html, markdown, json
                code_name, task_id = item['task_id'].split('/')
                code_file = os.path.join(data_dir, f'{code_name}', task_id+'.'+file_fix[code_name])
                # print(code_file)
                with open(code_file) as tf:
                    tlines = tf.readlines()
                    tlines = [l for l in tlines if len(l.strip())]
                    code = '\n'.join(tlines)
                item['canonical_solution'] = code 
                
                # continue 
            gen_codes = []
            for i in range(GEN_COUNT+50):
                mask_code = gen_lines_fim_data(item, mask_line_cnt=1)
                if mask_code is not None and mask_code not in gen_codes:
                    gen_codes.append(mask_code)
                if len(gen_codes) >= GEN_COUNT:
                    break 

            for idx, mask_code in enumerate(gen_codes):
                if 'signature' in item:
                    full_mask_code = item['signature']+'\n'+mask_code
                else:
                    full_mask_code = mask_code 
                
                instruction = PROMPT_TEMPLATE.format(lang=lang, docstring=item['docstring'] if 'docstring' in item else item['prompt'], code=full_mask_code)
                new_item = copy.deepcopy(item)
                new_item['instruction'] = instruction
                new_item['mask_code'] = mask_code 
                new_item['task_id'] += f'-{idx}-single'
                single_line_items.append(new_item)

        ############# mask multi line ###################
        for item in items:
            lang = item['task_id'].split('/')[0]

            gen_codes = []
            for i in range(GEN_COUNT+50):
                mask_code = gen_lines_fim_data(item, mask_line_cnt=random.randint(2, 5))
                if mask_code is not None and mask_code not in gen_codes:
                    gen_codes.append(mask_code)
                if len(gen_codes) >= GEN_COUNT:
                    break 
            for idx, mask_code in enumerate(gen_codes):
                if 'signature' in item:
                    full_mask_code = item['signature']+'\n'+mask_code
                else:
                    full_mask_code = mask_code 
                # instruction = PROMPT_TEMPLATE.format(lang=lang, docstring=item['docstring'], code=full_mask_code)
                instruction = PROMPT_TEMPLATE.format(lang=lang, docstring=item['docstring'] if 'docstring' in item else item['prompt'], code=full_mask_code)
                new_item = copy.deepcopy(item)
                new_item['instruction'] = instruction
                new_item['mask_code'] = mask_code 
                new_item['task_id'] += f'-{idx}-multi'
                
                multi_line_items.append(new_item)   
          
        ############# mask span ###################
        for item in items:
            lang = item['task_id'].split('/')[0]

            gen_codes = []
            for i in range(GEN_COUNT+50):
                mask_code = gen_span_fim_data(item)
                # print(mask_code)
                if mask_code is not None and mask_code not in gen_codes:
                    gen_codes.append(mask_code)
                if len(gen_codes) >= GEN_COUNT:
                    break 
            for idx, mask_code in enumerate(gen_codes):
                if 'signature' in item:
                    full_mask_code = item['signature']+'\n'+mask_code
                else:
                    full_mask_code = mask_code 
                # instruction = PROMPT_TEMPLATE.format(lang=lang, docstring=item['docstring'], code=full_mask_code)
                instruction = PROMPT_TEMPLATE.format(lang=lang, docstring=item['docstring'] if 'docstring' in item else item['prompt'], code=full_mask_code)
                new_item = copy.deepcopy(item)
                new_item['instruction'] = instruction
                new_item['mask_code'] = mask_code 
                new_item['task_id'] += f'-{idx}-span'
                rand_span_items.append(new_item)   

        ############# light span ###################
          
        for item in items:
            lang = item['task_id'].split('/')[0]

            tid = int(item['task_id'].split('/')[1])
            # if lang == 'JSON' and tid == 50:
            #     print(item['canonical_solution'])

            gen_codes = []
            for i in range(15):
                mask_code = gen_span_fim_data(item)
                # print(mask_code)
                if mask_code is not None and mask_code not in gen_codes:
                    gen_codes.append(mask_code)
                if len(gen_codes) == 1:
                    break 
            for idx, mask_code in enumerate(gen_codes):
                if 'signature' in item:
                    full_mask_code = item['signature']+'\n'+mask_code
                else:
                    full_mask_code = mask_code 
                instruction = PROMPT_TEMPLATE.format(lang=lang, docstring=item['docstring'] if 'docstring' in item else item['prompt'], code=full_mask_code)
                new_item = copy.deepcopy(item)
                new_item['instruction'] = instruction
                new_item['mask_code'] = mask_code 
                new_item['task_id'] += f'-{idx}-light-span'
                light_span_items.append(new_item)   

        for item in single_line_items:
            f_single.write(json.dumps(item, ensure_ascii=False)+ '\n')        
        for item in multi_line_items:
            f_multi.write(json.dumps(item, ensure_ascii=False)+ '\n')
        for item in rand_span_items:
            f_span.write(json.dumps(item, ensure_ascii=False)+ '\n')
        for item in light_span_items:
            f_light_span.write(json.dumps(item, ensure_ascii=False)+ '\n')

        for item in single_line_items+multi_line_items+rand_span_items:
            f_merge.write(json.dumps(item, ensure_ascii=False)+ '\n')
        f_single.close()
        f_multi.close()
        f_span.close()
        f_merge.close()
        print(f'{lang}: single:{len(single_line_items)} multi:{len(multi_line_items)} span: {len(rand_span_items)} light span: {len(light_span_items)}\n')
            
    # print(file_cnt)




