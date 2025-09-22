import torch
from PIL import Image
from .processors import get_image_string
from torch.utils.data import Dataset,IterableDataset
from utils.torch_utils import get_rank,get_world_size
import torch.distributed as dist
import random
random.seed(42)
import numpy as np

class BaseDataset(Dataset):

    def __init__(self,
        dataset, 
        tokenizer, 
        image_processor, 
        mp_image_token_length
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.user_prefix_len = self._get_prefix_len(role='user')
        self.assistant_prefix_len = self._get_prefix_len(role='assistant')

    
    def __len__(self):
        return len(self.dataset)
    
    def _get_prefix_len(self,role='assistant'):
        ''' role = assistant时，获得模板的前缀长度：<|im_start|>system...<|im_start|>assistant
            比如
            <|im_start|>system
            You are a helpful AI assistant named MxLLM, trained by MuXing<|im_end|>
            <|im_start|>assistant
            xzyvd<|im_end|>
            中
            <|im_start|>system
            You are a helpful AI assistant named MxLLM, trained by MuXing<|im_end|>
            <|im_start|>assistant
            的encode后的长度
        
        '''
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template(
            [{"role":role, "content":random_string_5_letters}],
            tokenize=False,
            add_special_tokens=False
        )
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))



    def _process_images(self,images):
        # 使用图像预处理器对图像进行预处理
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image, splitted_image_count =  self.image_processor(image)
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise TypeError(f"图像类型不是PIL")
        
        return processed_images,splitted_image_counts


    def _get_messages(self,item,splitted_image_counts):
        ''' 从每条数据中获取 图像的对话信息，如果输入的是多张大尺度图像，在最前面加上图像占位符字符串
        Args:
            item: 每条数据 {"images":[PIL], "texts":[{},{}]}
            splitted_image_counts: 图像根据 patch_size被切分后所得块的数量 [(n_h,n_w)]
        Return:
            messages = [
                {"role":"user", "content":image_string + "xxx1"},
                {"role":"assistant", "content":"yyy1"},
                {"role":"user", "content":"xxx2"},
                {"role":"assistant", "content":"yyy2"},
            ]
        '''

        messages = []
        for text in item['texts']:
            messages.append({"role":"user", "content":text["user"]})
            messages.append({"role":"assistant", "content":text["assistant"]})

        image_string = get_image_string(self.tokenizer,splitted_image_counts,self.mp_image_token_length)

        if len(splitted_image_counts) > 0:
            messages[0]["content"] = image_string + messages[0]["content"]
        
        return messages
    

    def _prepare_inputs_and_loss_mask(self,messages):
        """ 获得消息中的assistant回复的mask位置

            input_ids:      [
                1 9690 198 2683 359 253 5356 5646 11173 3365 372 104 16421 61 28 7018 411 14099 72 274 2 198 1 4093 198 44 720 79 33 79 2283 79 33 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 720 79 33 79 2283 79 34 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 720 79 34 79 2283 79 33 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 720 79 34 79 2283 79 34 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 21198 108 5028 108 46 1780 314 260 2380 282 260 22861 198 15939 357 5453 30 2 198 1 520 9531 198 15420 30 2 198]
            mask:           [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]
            attention_mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
]

        """
        # 将对话信息转为 ids
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids['input_ids'])

        cursor = 0
        for index,msg in enumerate(messages):
            segment_ids = self.tokenizer.apply_chat_template([msg],tokenize=True,add_special_tokens=False)
            seg_len = len(segment_ids)

            content_len = len(self.tokenizer.encode(msg["content"]))
            
            if index==0:
                if msg["role"] == "user":            
                    temp_len = len(self.tokenizer.encode("<|im_start|>user\n"))
                    system_len = self.user_prefix_len - temp_len
                else:
                    temp_len = len(self.tokenizer.encode("<|im_start|>assistant\n"))
                    system_len = self.assistant_prefix_len - temp_len
                cursor += system_len

            # 每句话的后缀长度 和 system长度
            if msg["role"] == "user":            
                temp_len = len(self.tokenizer.encode("<|im_start|>user\n"))
                suffix_len = seg_len - content_len - self.user_prefix_len
                cursor += temp_len + content_len + suffix_len
                
            else:
                temp_len = len(self.tokenizer.encode("<|im_start|>assistant\n"))
                suffix_len = seg_len - content_len - self.assistant_prefix_len
                start = cursor +temp_len
                cursor += temp_len + content_len + suffix_len
                end = cursor - suffix_len
                mask[start:end] = [1] * (end-start)
        
        return torch.tensor(conv_ids['input_ids']), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])
    

class VQADataset(BaseDataset):
    def _get_labels(self, input_ids, mask):

        labels = input_ids.clone().masked_fill(~mask,-100) # system user的token_id位置为-100,assistant位置值不变
        labels = labels.roll(-1)    # 整个序列向左滚动一位，实现“下一个 token 预测”任务
        labels[-1] = -100   #最后1个token不进行计算

        return labels
    
    def __getitem__(self,idx):
        item = self.dataset[idx]

        images_data = item['images']
        if not isinstance(images_data,list):
            images_data  = [images_data]
        processed_images,splitted_image_counts = self._process_images(images_data)
        messages = self._get_messages(item,splitted_image_counts)
        input_ids,mask,attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids,mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "idx":idx
        }



class ConstantLengthDataset(IterableDataset):
    """
    迭代式数据集，确保输出的每个batch都包含指定数量的有效样本
    适用于大数据场景，无需预加载全部数据
    """
    def __init__(self, base_dataset, max_length, batch_size,rank=None, world_size=None):
        self.base_dataset = base_dataset  # 原始数据集(VQADataset)
        self.max_length = max_length      # 最大序列长度限制
        self.batch_size = batch_size      # 目标batch大小
        self.rank = rank if rank is not None else get_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.base_iter = iter(self.base_dataset)
        
    def __iter__(self):
        batch = []
        while True:
            try:
                sample = next(self.base_iter)
            except StopIteration:
                self.base_iter = iter(self.base_dataset)
                continue
            
            if  len(sample["input_ids"])>0 and len(sample["input_ids"]) <= self.max_length:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    # print(f'iter rank {self.rank},{len(batch),[i['idx'] for i in batch]}')
                    yield batch
                    batch = []  # 清空，准备下一个 batch
            

    def __len__(self):
        return len(self.base_dataset) // self.batch_size


def synchronized_dataloader_step(train_loader, is_dist):
    """
    Create a synchronized iterator that handles uneven data distribution in DDP.
    All ranks will stop when the first rank runs out of data.
    This happens because when packing a presharded dataset, a rank might have less groups than the others.
    """
    if not is_dist:
        # For single GPU, we don't need synchronization.
        for batch in train_loader:
            yield batch
        return
    
    # For DDP, we need synchronization.
    train_iter = iter(train_loader)
    
    while True:
        try:
            batch = next(train_iter)
            has_data = torch.tensor(1, device=torch.cuda.current_device())
        except StopIteration:
            batch = None
            has_data = torch.tensor(0, device=torch.cuda.current_device())
        
        # We synchronize across all ranks. If any rank is out of data, all ranks stop.
        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
        
        if has_data.item() == 0:
            # At least one rank is out of data. All ranks should stop.
            break
        yield batch
    return None