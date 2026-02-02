
import os
import json
import torch
import clip
from PIL import Image
from PIL import ImageFile
from PIL import TiffImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader

class SentenceDataset(Dataset):
    def __init__(self, opt, data_path, photo_path, clip_model, preprocess):
        self.dataset_type = opt.dataset
        self.photo_path = photo_path
        
        self.clip_model = clip_model
        self.preprocess = preprocess
        

        with open(data_path, 'r', encoding='utf-8') as file_read:
            file_content = json.load(file_read)
        
        self.data_id_list = []
        self.text_list = []
        self.label_list = []
        for data in file_content:
            self.data_id_list.append(data['id'])
            self.text_list.append(data['text'])
            self.label_list.append(data['emotion_label'])
        
        if self.dataset_type != 'meme7k':
            self.image_id_list = [str(data_id) + '.jpg' for data_id in self.data_id_list]
        else:
            self.image_id_list = self.data_id_list
    
    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.data_id_list)
 
    def __getitem__(self, index):
        try:

            image_path = os.path.join(self.photo_path, str(self.data_id_list[index]) + '.jpg')
            

            if not os.path.exists(image_path):
                alternative_path = os.path.join('/kaggle/input/dataset-image2', 
                                              str(self.data_id_list[index]) + '.jpg')
                if os.path.exists(alternative_path):
                    image_path = alternative_path
                else:
                    raise FileNotFoundError(f"Image {self.data_id_list[index]} not found")
            

            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
            

            text = self.text_list[index]
            text = clip.tokenize(text, truncate=True)[0]
            
            return text, image, self.label_list[index]
            
        except Exception as e:
            print(f"Error processing item {index}: {e}")

            return self.__getitem__((index + 1) % self.__len__())

def data_process(opt, data_path, photo_path, data_type):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    dataset = SentenceDataset(opt, data_path, photo_path, clip_model, preprocess)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=opt.batch_size,
        shuffle=True if data_type == 1 else False,
        num_workers=opt.num_workers, 
        pin_memory=True
    )
    
    return data_loader, dataset.__len__()