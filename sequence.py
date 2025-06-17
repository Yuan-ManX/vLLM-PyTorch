from copy import copy
from enum import Enum, auto
from itertools import count

from sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举类，定义序列可能处于的不同状态
    
    Attributes:
        WAITING (SequenceStatus): 序列等待处理的状态
        RUNNING (SequenceStatus): 序列正在运行的状态
        FINISHED (SequenceStatus): 序列已完成的状态
    """
    WAITING = auto()   # 等待状态，表示序列尚未开始处理
    RUNNING = auto()   # 运行状态，表示序列正在处理中
    FINISHED = auto()  # 完成状态，表示序列处理已完成


class Sequence:
    """
    序列类，表示一个待处理的token序列
    
    该类封装了token序列的元数据、状态、采样参数以及与块管理相关的属性和方法
    """

    # 每个块包含的token数量，固定为256
    block_size = 256
    # 静态计数器，用于生成唯一的序列ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        """
        初始化序列对象
        
        Args:
            token_ids (List[int]): 初始的token ID列表
            sampling_params (SamplingParams): 采样参数对象，包含温度、最大token数等设置
        """
        # 获取下一个唯一的序列ID
        self.seq_id = next(Sequence.counter)
        # 初始化序列状态为等待
        self.status = SequenceStatus.WAITING
        # 深拷贝token ID列表，避免外部修改
        self.token_ids = copy(token_ids)
        # 记录最后一个token ID
        self.last_token = token_ids[-1]
        # 计算token总数
        self.num_tokens = len(self.token_ids)
        # 初始化提示token数量为总token数
        self.num_prompt_tokens = len(token_ids)
        # 初始化缓存的token数量为0
        self.num_cached_tokens = 0
        # 初始化块表为空列表
        self.block_table = []
        # 设置温度参数
        self.temperature = sampling_params.temperature
        # 设置最大token数参数
        self.max_tokens = sampling_params.max_tokens
        # 设置是否忽略结束符参数
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """
        返回序列的token数量
        
        Returns:
            int: token总数
        """
        return self.num_tokens

    def __lt__(self, other):
        """
        定义序列对象的小于比较操作，基于序列ID
        
        Args:
            other (Sequence): 另一个序列对象
        
        Returns:
            bool: 如果当前序列ID小于另一个序列ID则返回True，否则返回False
        """
        return self.seq_id < other.seq_id

    def __getitem__(self, key):
        """
        支持通过索引访问token ID
        
        Args:
            key (int): 索引
        
        Returns:
            int: 对应位置的token ID
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """
        属性：判断序列是否已完成
        
        Returns:
            bool: 如果状态为FINISHED则返回True，否则返回False
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        属性：计算完成部分的token数量
        
        Returns:
            int: 完成部分的token数量
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """
        属性：获取提示部分的token ID列表
        
        Returns:
            List[int]: 提示部分的token ID列表
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """
        属性：获取完成部分的token ID列表
        
        Returns:
            List[int]: 完成部分的token ID列表
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """
        属性：计算缓存的块数量
        
        Returns:
            int: 缓存的块数量
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        属性：计算序列所需的块总数
        
        Returns:
            int: 序列所需的块总数
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        属性：计算最后一个块中的token数量
        
        Returns:
            int: 最后一个块中的token数量
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取指定索引的块
        
        Args:
            i (int): 块索引
        
        Returns:
            List[int]: 对应块的token ID列表
        
        Raises:
            AssertionError: 如果索引超出范围
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        向序列追加一个token
        
        Args:
            token_id (int): 要追加的token ID
        """
        # 追加token ID到列表中
        self.token_ids.append(token_id)
        # 更新最后一个token ID
        self.last_token = token_id
        # 增加token总数
        self.num_tokens += 1

    def __getstate__(self):
        """
        定义序列对象的序列化行为
        
        当序列对象被pickle序列化时调用，移除token_ids以节省空间（如果已完成）
        
        Returns:
            dict: 序列对象的序列化状态
        """
        # 获取序列对象的属性字典
        state = vars(self).copy()
        if self.num_completion_tokens:
            # 如果有完成部分的token，则移除token_ids
            state.pop("token_ids")
        return state
