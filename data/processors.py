import torchvision.transforms as transforms
try:
    from .custom_transformers import DynamicResize, SplitImage
except ImportError:
    from custom_transformers import DynamicResize, SplitImage
from transformers import AutoTokenizer

# 图像预处理
def get_image_processor(max_img_size, splitted_img_size):
    return transforms.Compose([
        DynamicResize(splitted_img_size,max_img_size),
        transforms.ToTensor(),
        SplitImage(splitted_img_size)
        
    ])

TOKENIZERS_CACHE = {"HuggingFaceTB/SmolLM2-360M-Instruct":None}

# 获取分词模型
def get_tokenizer(name, extra_special_tokens=None, chat_template=None,cache_dir=None):

    tokenizer_init_kwargs = {"use_fast": True}
    if extra_special_tokens is not None:
        tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
    if chat_template is not None:
        tokenizer_init_kwargs["chat_template"] = chat_template
    if cache_dir is not None:
        tokenizer_init_kwargs["cache_dir"] = cache_dir

    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            **tokenizer_init_kwargs,
            )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            **tokenizer_init_kwargs,
            local_files_only=True,
            )
        
    TOKENIZERS_CACHE[name] = tokenizer
    
    return TOKENIZERS_CACHE[name]

def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    """获取图像的占位符字符串，知道图像在哪，每块用什么token表示---支持 高分辨率图像输入 和 空间感知理解
    Args:
        tokenizer: 分词器
        splitted_image_counts: 图像根据 patch_size被切分后所得块的数量 [(n_h,n_w)]
        mp_image_token_length: 64,
    Return:
        batch_size = 1时: 
            image_string = <row_1_col_1><|image|>...<|image|><row_1_col_2><|image|>...<|image|><row_2_col_1><|image|>...<|image|><row_2_col_2><|image|>...<|image|>
        batch_size > 1时: 
            image_string = <image: 0><row_1_col_1><|image|>...<|image|><row_1_col_2><|image|>...<|image|><row_2_col_1><|image|>...<|image|><row_2_col_2><|image|>...<|image|><image: 1><row_1_col_1><|image|>...<|image|>
    """

    image_string = ""
    for idx,(n_h,n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts)>1:
            image_string+=f"<image: {idx}>"
        
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f"r{i+1}c{j+1}") # "<row_4_col_2>"
                image_string += tokenizer.image_token * mp_image_token_length # "<|image|>"*64 = "<|image|><|image|>...<|image|>"

    return image_string


if __name__ == '__main__':

    lm_tokenizer = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_cache_dir = "/workspace/vlm/resources/models"
    lm_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    vlm_extra_tokens = {'image_token': '<|image|>', 'r1c1': '<row_1_col_1>', 'r1c2': '<row_1_col_2>', 'r1c3': '<row_1_col_3>', 'r1c4': '<row_1_col_4>', 'r2c1': '<row_2_col_1>', 'r2c2': '<row_2_col_2>', 'r2c3': '<row_2_col_3>', 'r2c4': '<row_2_col_4>', 'r3c1': '<row_3_col_1>', 'r3c2': '<row_3_col_2>', 'r3c3': '<row_3_col_3>', 'r3c4': '<row_3_col_4>', 'r4c1': '<row_4_col_1>', 'r4c2': '<row_4_col_2>', 'r4c3': '<row_4_col_3>', 'r4c4': '<row_4_col_4>'}
    a = get_tokenizer(lm_tokenizer, vlm_extra_tokens, lm_chat_template,lm_cache_dir)
    
    print(a)
    '''
    GPT2TokenizerFast(name_or_path='HuggingFaceTB/SmolLM2-360M-Instruct', vocab_size=49152, model_max_length=8192, is_fast=True, padding_side='right', truncation_side='right', 
    special_tokens={
        'bos_token': '<|im_start|>', 
        'eos_token': '<|im_end|>', 
        'pad_token': '<|im_end|>', 
        'unk_token': '<|endoftext|>', 
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>'], 'image_token': '<|image|>', 'r1c1': '<row_1_col_1>', 'r1c2': '<row_1_col_2>', 'r1c3': '<row_1_col_3>', 'r1c4': '<row_1_col_4>', 'r2c1': '<row_2_col_1>', 'r2c2': '<row_2_col_2>', 'r2c3': '<row_2_col_3>', 'r2c4': '<row_2_col_4>', 'r3c1': '<row_3_col_1>', 'r3c2': '<row_3_col_2>', 'r3c3': '<row_3_col_3>', 'r3c4': '<row_3_col_4>', 'r4c1': '<row_4_col_1>', 'r4c2': '<row_4_col_2>', 'r4c3': '<row_4_col_3>', 'r4c4': '<row_4_col_4>'}, 
        clean_up_tokenization_spaces=False, added_tokens_decoder={
            0: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            1: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            2: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            3: AddedToken("<repo_name>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            4: AddedToken("<reponame>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            5: AddedToken("<file_sep>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            6: AddedToken("<filename>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            7: AddedToken("<gh_stars>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            8: AddedToken("<issue_start>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            9: AddedToken("<issue_comment>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            10: AddedToken("<issue_closed>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            11: AddedToken("<jupyter_start>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            12: AddedToken("<jupyter_text>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            13: AddedToken("<jupyter_code>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            14: AddedToken("<jupyter_output>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            15: AddedToken("<jupyter_script>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            16: AddedToken("<empty_output>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49152: AddedToken("<|image|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49153: AddedToken("<row_1_col_1>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49154: AddedToken("<row_1_col_2>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49155: AddedToken("<row_1_col_3>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49156: AddedToken("<row_1_col_4>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49157: AddedToken("<row_2_col_1>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49158: AddedToken("<row_2_col_2>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49159: AddedToken("<row_2_col_3>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49160: AddedToken("<row_2_col_4>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49161: AddedToken("<row_3_col_1>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49162: AddedToken("<row_3_col_2>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49163: AddedToken("<row_3_col_3>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49164: AddedToken("<row_3_col_4>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49165: AddedToken("<row_4_col_1>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49166: AddedToken("<row_4_col_2>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49167: AddedToken("<row_4_col_3>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            49168: AddedToken("<row_4_col_4>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    }
    )
    
    '''