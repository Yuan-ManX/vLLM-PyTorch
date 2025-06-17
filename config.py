import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    模型配置类，用于定义和验证模型运行所需的各项参数
    
    该类封装了模型路径、硬件资源限制、并行策略等关键配置，
    并在初始化后进行必要的参数验证和配置加载
    """

    # 模型目录路径，指向预训练模型的本地存储位置
    model: str  
    # 批量处理的最大token数
    max_num_batched_tokens: int = 32768
    # 单个批量中允许的最大序列数量
    max_num_seqs: int = 512
    # 模型支持的最大序列长度
    max_model_len: int = 4096
    # GPU内存利用率
    gpu_memory_utilization: float = 0.9
    # 张量并行度
    tensor_parallel_size: int = 1
    # 是否启用即时执行模式
    enforce_eager: bool = False
    # Hugging Face模型配置对象
    hf_config: AutoConfig | None = None
    # 结束符token的ID
    eos: int = -1
    # KV缓存块大小
    kvcache_block_size: int = 256
    # KV缓存块数量
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """
        数据类初始化后的后处理方法
        
        在类实例化完成后自动调用，用于验证参数合法性
        并加载预训练模型的配置
        """

        # 验证模型路径是否为有效目录
        assert os.path.isdir(self.model)
        # 验证kvcache_block_size是否为256的倍数
        assert self.kvcache_block_size % 256 == 0
        # 验证tensor_parallel_size是否在1到8之间
        assert 1 <= self.tensor_parallel_size <= 8
        # 加载预训练模型的配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 确保max_model_len不超过模型支持的最大位置嵌入数
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
