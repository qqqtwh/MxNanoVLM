import torch.nn as nn

class ModalityProjector(nn.Module):

    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.scale_factor = cfg.mp_pixel_shuffle_factor # 4
        self.input_dim = cfg.vit_hidden_dim * (self.scale_factor**2) # 768*4*4
        self.output_dim = cfg.lm_hidden_dim # 960

        self.proj = nn.Linear(self.input_dim,self.output_dim,bias=False) # (bs,768*4*4,960)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def pixel_shuffle(self,x):
        """ 将图像的pathch块重排提取特征, 维度转换如下:
            x = (bs, 64, 768)       # bs, seq, dim
            h_out = w_out = 8/4 
            x = reshape(bs,2,4,2,4,768)    # bs, h_out, scale_factor, w_out, scale_factor, dim
            x = permute(bs,2,2,4,4,768)    # bs, h_out, w_out, scale_factor, scale_factor, dim
            x = reshape(bs,2*2,768*4*4)    # bsz, h_out * w_out, embed_dim * self.scale_factor**2

            输入: 8x8 个 patch,每个 768 维         ---->         输出: 2x2 个patch, 每个 768*4*4 维
            ┌───┬───┬───┬───┬───┬───┬───┬───┐
            │01 │02 │03 │04 │05 │06 │07 │08 │
            ├───┼───┼───┼───┼───┼───┼───┼───┤
            │09 │10 │11 │12 │13 │14 │15 │16 │
            ├───┼───┼───┼───┼───┼───┼───┼───┤           ┌────────────────────────────┬────────────────────────────┐
            │17 │18 │19 │20 │21 │22 │23 │24 │           │                  Patch 1   │ Patch 2                    │
            ├───┼───┼───┼───┼───┼───┼───┼───┤           │   01,02,03,04; 09,10,11,12 │ 05,06,07,08; 13,14,15,16   │
            │25 │26 │27 │28 │29 │30 │31 │32 │           │   17,18,19,20; 25,26,27,28 │ 21,22,23,24; 29,30,31,32   │
            ├───┼───┼───┼───┼───┼───┼───┼───┤   ---->   ├────────────────────────────┼────────────────────────────┤
            │33 │34 │35 │36 │37 │38 │39 │40 │           │                    Patch 3 │ Patch 4                    │
            ├───┼───┼───┼───┼───┼───┼───┼───┤           │   33,34,35,36; 41,42,43,44 │ 37,38,39,40; 45,46,47,48   │
            │41 │42 │43 │44 │45 │46 │47 │48 │           │   49,50,51,52; 57,58,59,60 │ 53,54,55,56; 61,62,63,64   │
            ├───┼───┼───┼───┼───┼───┼───┼───┤           └────────────────────────────┴────────────────────────────┘
            │49 │50 │51 │52 │53 │54 │55 │56 │
            ├───┼───┼───┼───┼───┼───┼───┼───┤
            │57 │58 │59 │60 │61 │62 │63 │64 │
            └───┴───┴───┴───┴───┴───┴───┴───┘

        """
        bsz, seq, embed_dim = x.size()  # bs,32*32,768
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Seq 长度必须可以开平方
        assert seq_root % self.scale_factor == 0 # Sequence root 必须是scale_factor的倍数

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)   # (bs, 32*32, 768)->(bs, 32, 32, 768) 
        h_out = height // self.scale_factor # 8
        w_out = width // self.scale_factor  # 8
        
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2) # (bs, 8*8, 768*4*4)
        
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)   # (bs, 32*32, 768) -> (bs, (32//4)**2, 768*4*4)
        x = self.proj(x)            # (bs, (32//4)**2, 768*4*4) -> (bs, (32//4)**2, 960)

        return x