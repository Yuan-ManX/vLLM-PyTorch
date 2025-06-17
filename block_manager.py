from collections import deque
import xxhash
import numpy as np

from sequence import Sequence


def compute_hash(token_ids: list[int], prefix: int = -1):
    """
    计算给定token序列的哈希值
    
    如果提供了前缀，则将前缀的字节表示与token序列的字节表示拼接后进行哈希计算
    否则，仅对token序列的字节表示进行哈希计算
    
    Args:
        token_ids (List[int]): 要计算哈希的token ID列表
        prefix (int, optional): 前缀整数，默认为-1（表示不使用前缀）
    
    Returns:
        int: 计算得到的哈希值
    """
    # 初始化xxhash的64位哈希对象
    h = xxhash.xxh64()
    if prefix != -1:
        # 如果提供了前缀，将其转换为8字节小端字节序并更新到哈希对象中
        h.update(prefix.to_bytes(8, "little"))
    # 将token ID列表转换为字节数组并更新到哈希对象中
    h.update(np.array(token_ids).tobytes())
    # 返回最终的哈希整数值
    return h.intdigest()


class Block:
    """
    块类，表示存储token序列的单个块
    
    每个块都有一个唯一的ID、引用计数、哈希值和存储的token ID列表
    """

    def __init__(self, block_id):
        # 块的唯一ID
        self.block_id = block_id
        # 引用计数，记录当前块被引用的次数
        self.ref_count = 0
        # 块的哈希值，-1表示未设置
        self.hash = -1
        # 存储在块中的token ID列表
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的内容和哈希值
        
        Args:
            hash (int): 新的哈希值
            token_ids (List[int]): 新的token ID列表
        
        Raises:
            AssertionError: 如果提供的哈希值为-1
        """
        assert hash != -1
        # 更新块的哈希值
        self.hash = hash
        # 更新块的token ID列表
        self.token_ids = token_ids

    def reset(self):
        """
        重置块的状态
        
        将引用计数设置为1，哈希值重置为-1，token ID列表清空
        """
        # 重置引用计数为1
        self.ref_count = 1
        # 重置哈希值为-1
        self.hash = -1
        # 清空token ID列表
        self.token_ids = []

    def __repr__(self):
        """
        返回块的字符串表示
        
        Returns:
            str: 包含块ID、引用计数和哈希值的字符串
        """
        return f"{(self.block_id, self.ref_count, self.hash)}"


class BlockManager:
    """
    块管理器类，用于管理多个块的分配和回收
    
    通过维护空闲块列表和已使用块列表，实现高效的块分配和缓存机制
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器
        
        Args:
            num_blocks (int): 总块数
            block_size (int): 每个块的最大token数量
        """
        # 每个块的最大token数量
        self.block_size = block_size
        # 初始化块列表，每个块是一个Block对象
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希值到块ID的映射字典，用于快速查找块
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块ID的双端队列，用于高效地分配和回收块
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用块ID的集合，用于快速检查块是否被使用
        self.used_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int):
        """
        分配一个空闲块
        
        将指定的块从空闲列表中移除，并添加到已使用列表中
        重置块的状态以准备新的数据存储
        
        Args:
            block_id (int): 要分配的块ID
        
        Returns:
            Block: 被分配的块对象
        
        Raises:
            AssertionError: 如果块的引用计数不为0
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 重置块状态
        block.reset()
        # 从空闲列表中移除
        self.free_block_ids.remove(block_id)
        # 添加到已使用列表中
        self.used_block_ids.add(block_id)
        # 返回被分配的块对象
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        """
        回收一个已使用块
        
        将指定的块从已使用列表中移除，并添加到空闲列表中
        
        Args:
            block_id (int): 要回收的块ID
        
        Raises:
            AssertionError: 如果块的引用计数不为0
        """
        assert self.blocks[block_id].ref_count == 0
        # 从已使用列表中移除
        self.used_block_ids.remove(block_id)
        # 添加到空闲列表中
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence):
        """
        检查是否可以为序列分配足够的块
        
        Args:
            seq (Sequence): 要检查的序列对象
        
        Returns:
            bool: 如果有足够的空闲块则返回True，否则返回False
        """
        # 比较空闲块数量与序列所需的块数量
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配块并更新其块表
        
        如果缓存命中，则重用现有块；否则，分配新块并更新哈希映射
        
        Args:
            seq (Sequence): 要分配块的序列对象
        
        Raises:
            AssertionError: 如果序列的块表不为空
        """
        assert not seq.block_table
        # 初始化哈希值为-1
        h = -1
        # 标记是否缓存未命中
        cache_miss = False

        for i in range(seq.num_blocks):
            # 获取第i个块的token ID列表
            token_ids = seq.block(i)
            # 如果块大小符合预期，则计算哈希值
            # 否则，重置哈希值为-1
            h = compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 查找哈希值对应的块ID
            block_id = self.hash_to_block_id.get(h, -1)

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 如果未找到或token ID不匹配，则标记缓存未命中
                cache_miss = True
            if cache_miss:
                # 从空闲列表中分配一个新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 增加缓存的token数量
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    # 增加引用计数
                    block.ref_count += 1
                else:
                    # 分配新块
                    block = self._allocate_block(block_id)
            if h != -1:
                # 更新块的内容和哈希值
                block.update(h, token_ids)
                # 更新哈希到块ID的映射
                self.hash_to_block_id[h] = block_id
            # 将块ID添加到序列的块表中
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        回收序列占用的所有块
        
        减少每个块的引用计数，如果引用计数为0，则回收块
        
        Args:
            seq (Sequence): 要回收块的序列对象
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            # 减少引用计数
            block.ref_count -= 1
            if block.ref_count == 0:
                # 回收块
                self._deallocate_block(block_id)
        # 重置缓存的token数量
        seq.num_cached_tokens = 0
        # 清空序列的块表
        seq.block_table.clear()

    def can_append(self, seq: Sequence):
        """
        检查是否可以向序列追加数据
        
        Args:
            seq (Sequence): 要检查的序列对象
        
        Returns:
            bool: 如果有足够的空闲块则返回True，否则返回False
        """
        # 检查空闲块数量是否满足追加条件
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        根据当前状态决定是否向序列追加数据
        
        如果追加后需要分配新块，则进行分配；否则，更新最后一个块的哈希值
        
        Args:
            seq (Sequence): 要追加数据的序列对象
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 确保最后一个块的哈希值已设置
            assert last_block.hash != -1
            # 分配一个新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 确保最后一个块的哈希值未设置
            assert last_block.hash == -1
            # 获取当前块的token ID列表
            token_ids = seq.block(seq.num_blocks-1)
            # 获取前一个块的哈希值作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            # 计算哈希值
            h = compute_hash(token_ids, prefix)
            # 更新最后一个块的内容和哈希值
            last_block.update(h, token_ids)
            # 更新哈希到块ID的映射
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 确保最后一个块的哈希值未设置
            assert last_block.hash == -1
