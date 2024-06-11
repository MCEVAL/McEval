import os 
import json

template1 = '''Write a {lang} function `{signature}` to solve the following problem:\n{docstring}'''
template_awk  = '''Using the awk command in Linux, complete the following task:\n{docstring}'''
template_html = '''Generate HTML code according to the following requirements:\n{docstring}'''
template_json = '''Create a JSON object according to the following requirements:\n{docstring}'''
template_md   = '''Generate Markdown code according to the following requirements:\n{docstring}'''


def gen_stage2_instruction(item):
    item['stage1_instruction'] = item['instruction']
    item['stage1_raw_generation'] = item['raw_generation']
    
    lang = item['task_id'].split('/')[0].strip()

    if lang.lower() == 'awk':
        item['instruction'] = template_awk.format(docstring=item['raw_generation'][0])
    elif lang.lower()== 'html':
        item['instruction'] = template_html.format(docstring=item['raw_generation'][0])
    elif lang.lower()== 'json':
        item['instruction'] = template_json.format(docstring=item['raw_generation'][0])
    elif lang.lower()== 'markdown':
        item['instruction'] = template_md.format(docstring=item['raw_generation'][0])
    else:    
        item['instruction'] = template1.format(lang=lang, signature=item['signature'], docstring=item['raw_generation'][0])
    
    del item['raw_generation']

if __name__ == '__main__':
    result_dir = './explain_stage1'
    output_dir = './explain_stage2'
    stage1_result_files = os.listdir(result_dir)
    
    for stage1_result_file in stage1_result_files:
        json_path = os.path.join(result_dir, stage1_result_file)
        if not os.path.exists(json_path):
            continue 
        items = [json.loads(x) for x in open(json_path).readlines() if x]
        for item in items:
            gen_stage2_instruction(item)

        save_dir = os.path.join(output_dir, stage1_result_file.replace('.jsonl', ''))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, stage1_result_file)    
        with open(save_path, 'w') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
                
