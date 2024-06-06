import os 
import json 



if __name__ == '__main__':
    save_dir = '/workspace/explain_data'
    # files = os.listdir('./data')
    data_dir = '/workspace/MMCodeEval/data'
    files = os.listdir(data_dir)
    files = [x for x in files if x.endswith('.jsonl')]

    file_fix = {'HTML':'html','Markdown':'md','JSON':'json'}
    # file_cnt = 0
    for file in files:
        
        file_path = os.path.join(data_dir, file)
        items = [json.loads(x) for x in open(file_path).readlines() if x]
        # print(file_path, len(items))
        # file_cnt+=1
        save_path = os.path.join(save_dir, file)
        f =  open(save_path,'w')
        for item in items:
            lang = item['task_id'].split('/')[0]
            item['instruction'] = f"Provide a concise natural language description (docstring) of the {lang} code in English using at most 500 characters."
            
            if 'signature' in item and 'canonical_solution' in item:
                code = item['signature'] +'\n' +item['canonical_solution']

            elif 'canonical_solution' not in item: # html, markdown, json
                code_name, task_id = item['task_id'].split('/')
                code_file = f'/workspace/MMCodeEval/data/{code_name}/'+task_id+'.'+file_fix[code_name]
                with open(code_file) as tf:
                    tlines = tf.readlines()
                    tlines = [l for l in tlines if len(l.strip())]
                    code = '\n'.join(tlines)

            else: #  awk
                # print(item['task_id'])
                code = item['canonical_solution']
        
            # print(item['task_id'])
            # print(code)
            # print('='* 100)
            item['instruction'] = code+'\n\n'+item['instruction']
        
            f.write(json.dumps(item, ensure_ascii=False)+ '\n')
        f.close()
  
    


