


if __name__ == '__main__':
    from data.processors import get_tokenizer

    lm_tokenizer = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = get_tokenizer(lm_tokenizer,cache_dir='/workspace/vlm/resources/models')
    random_string_5_letters = "xzyvd"

    # assistant 前缀
    assistant_random_string_chat_templated = tokenizer.apply_chat_template(
        [{"role":"assistant", "content":random_string_5_letters}],
        tokenize=False,
        add_special_tokens=False
    )
    assistant_random_string_location = assistant_random_string_chat_templated.find(random_string_5_letters)
    assistant_prefix = tokenizer.encode(assistant_random_string_chat_templated[:assistant_random_string_location])
    
    # role 前缀
    role_random_string_chat_templated = tokenizer.apply_chat_template(
        [{"role":"user", "content":random_string_5_letters}],
        tokenize=False,
        add_special_tokens=False
    )
    role_random_string_location = role_random_string_chat_templated.find(random_string_5_letters)
    role_prefix = tokenizer.encode(role_random_string_chat_templated[:role_random_string_location])
    
    role_prefix_len = len(role_prefix)
    assistant_prefix_len = len(assistant_prefix)
    
    
    
    print('--')

    
    messages = [
            {"role":"user", "content": "xxx1"},
            {"role":"assistant", "content":"yyy1"},
            {"role":"user", "content":"xxx2"},
            {"role":"assistant", "content":"yyy2"},
            {"role":"assistant", "content":"yyy3"},
        ]
    conv_ids = tokenizer.apply_chat_template(messages,tokenize=True,return_dict=True,add_special_tokens=False)
    mask = [0]*len(conv_ids['input_ids'])
    cursor = 0
    print('or_mask',mask,len(mask))

    for index,msg in enumerate(messages):
        segment_ids = tokenizer.apply_chat_template([msg],tokenize=True,add_special_tokens=False)
        seg_len = len(segment_ids)

        content_len = len(tokenizer.encode(msg["content"]))
        
        if index==0:
            if msg["role"] == "user":            
                temp_len = len(tokenizer.encode("<|im_start|>user\n"))
                system_len = role_prefix_len - temp_len
            else:
                temp_len = len(tokenizer.encode("<|im_start|>assistant\n"))
                system_len = assistant_prefix_len - temp_len
            cursor += system_len

        # 每句话的后缀长度 和 system长度
        if msg["role"] == "user":            
            temp_len = len(tokenizer.encode("<|im_start|>user\n"))
            suffix_len = seg_len - content_len - role_prefix_len
            cursor += temp_len + content_len + suffix_len
            
        else:
            temp_len = len(tokenizer.encode("<|im_start|>assistant\n"))
            suffix_len = seg_len - content_len - assistant_prefix_len
            start = cursor +temp_len
            cursor += temp_len + content_len + suffix_len
            end = cursor - suffix_len
            mask[start:end] = [1] * (end-start)
            
    '''
        # 每一句完整 25 26
        user 29 [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 372, 104, 16421, 61, 28, 7018, 411, 14099, 72, 274, 2, 198,    1, 4093, 198,          -- 46285, 33,--        2, 198]
        assi 31 [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 372, 104, 16421, 61, 28, 7018, 411, 14099, 72, 274, 2, 198,    1, 520, 9531, 198,     --27413, 105, 33,--    2, 198]
        user 29 [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 372, 104, 16421, 61, 28, 7018, 411, 14099, 72, 274, 2, 198,    1, 4093, 198,          --46285, 34,--         2, 198]
        assi 31 [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 372, 104, 16421, 61, 28, 7018, 411, 14099, 72, 274, 2, 198,    1, 520, 9531, 198,     --27413, 105, 34,--    2, 198]

        0 .... 18 19 20 21      22 23 24 25 26 27 28      29 30 31 32 33 34 35 36 37     38 39 40 41 42 43 44      45 46 47 48 49 50 51 52 53
        0 ....          0        0  0  0  0 0  0   0       0  0  0  0  1  1 1  0  0       0  0  0  0  0 0  0        0  0  0  0  1  1  1  0  0

        <|im_start|>system\nYou are a ...<|im_end|>\n
        1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 372, 104, 16421, 61, 28, 7018, 411, 14099, 72, 274, 2, 198,  ==== 22
        
        <|im_start|>user\nxxx1<|im_end|>\n
        1, 4093, 198,     ---46285, 33,--- 2, 198,
        
        <|im_start|>assistant\nyyy1<|im_end|>\n
        1, 520, 9531, 198,      ---27413, 105, 33,---         2, 198, 
        
        <|im_start|>user\nxxx2<|im_end|>\n
        1, 4093, 198, ---46285, 34---, 2, 198, 
        
        <|im_start|>assistant\nyyy2<|im_end|>\n
        1, 520, 9531, 198,      ---27413, 105, 34,---         2, 198
    '''
    
    print('ho_mask',mask,len(mask))
    '''


    
    

    
    
    '''