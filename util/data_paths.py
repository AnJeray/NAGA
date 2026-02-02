# util/data_paths.py

def prepare_data_paths(opt):
    dataset = opt.dataset
    data_path_root = {
        'train': f'dataset/{dataset}/train.json',
        'dev': f'dataset/{dataset}/dev.json',
        'test': f'dataset/{dataset}/test.json'
    }
    
    photo_paths = {
        'HFM': 'input/datasethfm/HFM',
        'MVSA_multiple': 'input/mvsamultiple/MVSA/data',
        'MVSA_Single': 'input/mvsasingle/MVSA_Single/data'
    }
    
    photo_path = photo_paths.get(dataset, photo_paths['MVSA_Single'])
    
    return data_path_root, photo_path