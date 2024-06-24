import os 
import json 
from collections import defaultdict

import argparse

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--split_file", default="", type=str, help="inference results to be splitted")
    parser.add_argument("--save_dir", default="", type=str, help="The folder where the split results are saved")
    args = parser.parse_args()
    # load_dir = './'
    save_dir = args.save_dir
    path = args.split_file
    filename = os.path.basename(path)

    lang_item_dict = defaultdict(list)
    items = []
    
    items+=[json.loads(x) for x in open(path).readlines() if len(x) > 0]
    
    for item in items:
        task_id = item['task_id']
        lang = task_id.split('/')[0].strip()
        # if 'base' in filename and lang not in ['JSON', 'AWK', 'HTML', 'Markdown']:
        #     if item["signature"] not in item["raw_generation"][0]:
        #         item["raw_generation"][0] = item['prompt'] + item["raw_generation"][0]
        lang_item_dict[lang].append(item)
    for lang, lang_items in lang_item_dict.items():
        # lang_items.sort(key=lambda x: int(x['task_id'].split('/')[1]))
        if not os.path.exists(os.path.join(save_dir, filename.replace('.jsonl', ''))):
            os.mkdir(os.path.join(save_dir, filename.replace('.jsonl', '')))
        save_file = os.path.join(save_dir, filename.replace('.jsonl', ''), lang+'.jsonl')
        
        with open(save_file, 'w') as f:
            for item in lang_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

