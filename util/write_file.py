import os
import random

def WriteFile(file_dir, file_name, file_content, file_mode, change_file_name=False):

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    elif os.path.exists(file_dir) and change_file_name:
        while os.path.exists(file_dir):         
            file_dir_list = file_dir.split('-')
            file_dir_list[6] = str((int(file_dir_list[6]) + random.randint(61, 119)) % 60)
            if len(file_dir_list[6]) == 1:
                file_dir_list[6] = '0' + file_dir_list[6]
            file_dir = '-'.join(file_dir_list)
        os.mkdir(file_dir)
    

    with open(os.path.join(file_dir, file_name), file_mode, encoding='utf-8') as f:
        f.write(file_content)
    
    return file_dir