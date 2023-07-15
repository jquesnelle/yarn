from transformers.models.llama.configuration_llama import LlamaConfig as LlamaConfigOriginal

class LlamaConfig(LlamaConfigOriginal):
    def __init__(self, use_xpos=False, position_interpolation_scale=1, ntk_alpha=None, transformer_engine=None, part_ntk_scale=None,use_sparse=False, use_block=False, use_torch=True, use_flash=False, sparsity=None, **kwargs):
        self.use_xpos = use_xpos
        self.position_interpolation_scale = position_interpolation_scale
        self.transformer_engine = transformer_engine
        self.ntk_alpha = ntk_alpha
        self.part_ntk_scale = part_ntk_scale
        self.use_sparse = use_sparse
        self.use_block = use_block
        self.use_torch = use_torch
        self.use_flash = use_flash
        self.sparsity = sparsity
        super().__init__(**kwargs)
