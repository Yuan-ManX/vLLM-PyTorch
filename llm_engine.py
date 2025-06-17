import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
import torch.multiprocessing as mp

from config import Config
from sampling_params import SamplingParams
from sequence import Sequence
from scheduler import Scheduler
from model_runner import ModelRunner


class LLMEngine:
    """
    大语言模型引擎类，用于管理模型运行、序列调度和生成文本
    
    该类集成了模型配置、并行处理、序列调度和文本生成功能，
    提供高层次的API用于生成文本
    """

    def __init__(self, model, **kwargs):
        """
        初始化LLMEngine对象
        
        Args:
            model (str): 预训练模型的路径或名称
            **kwargs: 其他配置参数，如max_num_batched_tokens, max_num_seqs等
        
        初始化过程包括：
        1. 从Config类中提取配置字段
        2. 根据传入的kwargs构建Config对象
        3. 初始化多进程和事件对象，用于并行处理
        4. 启动模型运行器进程
        5. 初始化tokenizer和scheduler
        6. 注册退出时的清理函数
        """
        # 从Config类中提取所有字段名称
        config_fileds = {field.name for field in fields(Config)}
        # 从kwargs中筛选出属于Config类的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}
        # 构建Config对象
        config = Config(model, **config_kwargs)
        # 初始化进程和事件列表
        self.ps = []
        self.events = []
        # 启动多个模型运行器进程，实现并行处理
        for i in range(1, config.tensor_parallel_size):
            # 创建一个事件对象，用于进程间通信
            event = mp.Event()
            # 创建并启动一个模型运行器进程
            process = mp.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            # 将进程对象添加到列表中
            self.ps.append(process)
            # 将事件对象添加到列表中
            self.events.append(event)
        
        # 初始化主模型运行器进程，进程ID为0
        self.model_runner = ModelRunner(config, 0, self.events)
        # 初始化tokenizer，加载预训练模型的词汇表
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 设置Config对象中的eos参数为tokenizer的eos_token_id
        config.eos = self.tokenizer.eos_token_id
        # 初始化调度器对象
        self.scheduler = Scheduler(config)
        # 注册退出时的清理函数，确保进程正确终止
        atexit.register(self.exit)

    def exit(self):
        """
        退出时的清理方法
        
        该方法在程序退出时被调用，确保所有子进程被正确终止
        """
        # 通知主模型运行器进程退出
        self.model_runner.call("exit")
        # 删除主模型运行器对象
        del self.model_runner
        # 等待所有子进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个生成请求到调度器
        
        Args:
            prompt (str | list[int]): 输入的提示，可以是字符串或token ID列表
            sampling_params (SamplingParams): 采样参数对象，包含温度、最大token数等设置
        
        如果提示是字符串，则使用tokenizer进行编码
        然后创建一个Sequence对象并添加到调度器中
        """
        if isinstance(prompt, str):
            # 如果提示是字符串，则使用tokenizer进行编码
            prompt = self.tokenizer.encode(prompt)
        # 创建一个Sequence对象
        seq = Sequence(prompt, sampling_params)
        # 将Sequence对象添加到调度器中
        self.scheduler.add(seq)

    def step(self):
        """
        执行模型运行和序列调度的一步
        
        Returns:
            Tuple[List[Tuple[int, List[int]]], int]: 
                - 第一个元素是完成的序列列表，每个元素是一个元组(序列ID, token ID列表)
                - 第二个元素是生成的token数量（如果是预填充阶段则为正数，解码阶段为负数）
        """
        # 从调度器中获取需要运行的序列和是否处于预填充阶段的标志
        seqs, is_prefill = self.scheduler.schedule()
        # 调用模型运行器执行模型前向传播，获取生成的token ID列表
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 对生成的token ID列表进行后处理，更新序列状态等
        self.scheduler.postprocess(seqs, token_ids)
        # 收集完成的序列及其token ID列表
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 计算生成的token数量
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """
        检查所有序列是否都已完成
        
        Returns:
            bool: 如果所有序列都已完成则返回True，否则返回False
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        生成文本的顶层接口
        
        Args:
            prompts (list[str] | list[list[int]]): 输入的提示列表，可以是字符串或token ID列表
            sampling_params (SamplingParams | list[SamplingParams]): 采样参数，可以是单个对象或列表
            use_tqdm (bool): 是否使用进度条，默认为True
        
        Returns:
            list[dict]: 生成的文本列表，每个元素包含"text"和"token_ids"键
        """
        if use_tqdm:
            # 初始化进度条
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True,
            )
        
        # 如果sampling_params不是列表，则将其转换为列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 将所有提示添加到调度器中
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 初始化输出字典
        outputs = {}
        # 初始化吞吐量变量
        prefill_throughput = decode_throughput = 0.
        # 主循环，直到所有序列完成
        while not self.is_finished():
            # 记录当前时间
            t = perf_counter()
            # 执行一步
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    # 计算预填充阶段的吞吐量
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # 计算解码阶段的吞吐量
                    decode_throughput = -num_tokens / (perf_counter() - t)
                # 更新进度条的后缀信息
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 处理生成的输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        # 按序列ID排序输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        # 解码token ID列表为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            # 关闭进度条
            pbar.close()
        return outputs
