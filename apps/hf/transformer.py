from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from transformers.models.mistral import MistralForCausalLM, MistralConfig
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Config

from lingua.transformer import BaseTransformerArgs


@dataclass
class LMTransformerFromHFArgs(BaseTransformerArgs):
    max_seqlen: int = -1
    vocab_size: int = -1
    dim: int = -1
    n_layers: int = -1

    seed: int = 42
    

def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


@dataclass
class HFCausalLMArgs:
    model_name: str = "openai-community/gpt2"
    # dict of model specific arguments
    model_args: Optional[Dict[str, Any]] = None


def get_hf_model(args: HFCausalLMArgs, seed: int = 0) -> Tuple[PreTrainedModel, LMTransformerFromHFArgs]:
    if "llama" in args.model_name:
        hf_conf = LlamaConfig.from_pretrained(args.model_name, **args.model_args)
        hf_model = LlamaForCausalLM(hf_conf)
        transformer_args = LMTransformerFromHFArgs(
            max_seqlen=hf_conf.max_position_embeddings,
            vocab_size=hf_conf.vocab_size,
            dim=hf_conf.hidden_size,
            n_layers=hf_conf.num_hidden_layers,
            seed=seed,
        )
    elif "mistral" in args.model_name:
        hf_conf = MistralConfig.from_pretrained(args.model_name, **args.model_args)
        hf_model = MistralForCausalLM(hf_conf)
        transformer_args = LMTransformerFromHFArgs(
            max_seqlen=hf_conf.max_position_embeddings,
            vocab_size=hf_conf.vocab_size,
            dim=hf_conf.hidden_size,
            n_layers=hf_conf.num_hidden_layers,
            seed=seed,
        )
    elif "gpt2" in args.model_name:
        hf_conf = GPT2Config.from_pretrained(args.model_name, **args.model_args)
        hf_model = GPT2LMHeadModel(hf_conf)
        transformer_args = LMTransformerFromHFArgs(
            max_seqlen=hf_conf.n_positions,
            vocab_size=hf_conf.vocab_size,
            dim=hf_conf.n_embd,
            n_layers=hf_conf.n_layer,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    
    return hf_model, transformer_args


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args):
    raise NotImplementedError("build_fsdp_grouping_plan is not implemented yet. For HF models, we recommend using default_fsdp_grouping_plan.")


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args, distributed_args):
    raise NotImplementedError("tp_parallelize is not implemented yet")