# src/anee/model_adapter.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    GPTJForCausalLM,
)

################################################################################
# GPT-2 FAMILY
################################################################################

def _adapt_gpt2(model):
    tf = model.transformer

    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.h),
        "embed_tokens": tf.wte,
        "pos_embed": tf.wpe,
        "final_ln": tf.ln_f,
        "lm_head": model.lm_head,
        "arch": "gpt2",
    }

################################################################################
# GPT-Neo
################################################################################

def _adapt_gptneo(model):
    tf = model.transformer

    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.h),
        "embed_tokens": tf.wte,
        "pos_embed": tf.wpe,
        "final_ln": tf.ln_f,
        "lm_head": model.lm_head,
        "arch": "gptneo",
    }

################################################################################
# GPT-J
################################################################################

def _adapt_gptj(model):
    tf = model.transformer

    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.h),
        "embed_tokens": tf.wte,
        "pos_embed": tf.wpe,
        "final_ln": tf.ln_f,
        "lm_head": model.lm_head,
        "arch": "gptj",
    }

################################################################################
# LLaMA / LLaMA-2 / LLaMA-3
################################################################################

def _adapt_llama(model):
    tf = model.model

    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.layers),
        "embed_tokens": tf.embed_tokens,
        "pos_embed": lambda ids: None,  # RoPE handled inside blocks
        "final_ln": tf.norm,
        "lm_head": model.lm_head,
        "arch": "llama",
    }

################################################################################
# Falcon 7B / 40B  (tiiuae/falcon-*)
################################################################################

def _adapt_falcon(model):
    tf = model.transformer

    # Falcon uses "h" for blocks, but attention is slightly different.
    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.h),
        "embed_tokens": tf.word_embeddings,
        "pos_embed": lambda ids: None,  # Falcon uses ALiBi, added inside attention
        "final_ln": tf.ln_f,
        "lm_head": model.lm_head,
        "arch": "falcon",
    }

################################################################################
# Mistral 7B / Mixtral 8x7B
################################################################################

def _adapt_mistral(model):
    tf = model.model

    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.layers),
        "embed_tokens": tf.embed_tokens,
        "pos_embed": lambda ids: None,  # RoPE
        "final_ln": tf.norm,
        "lm_head": model.lm_head,
        "arch": "mistral",
    }

################################################################################
# Phi-2 (Microsoft)
################################################################################

def _adapt_phi2(model):
    tf = model.model

    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.layers),
        "embed_tokens": tf.embed_tokens,
        "pos_embed": lambda ids: None,
        "final_ln": tf.final_layernorm,
        "lm_head": model.lm_head,
        "arch": "phi2",
    }

################################################################################
# Qwen-2 (Alibaba)
################################################################################

def _adapt_qwen2(model):
    tf = model.model

    return {
        "model": model,
        "config": model.config,
        "layers": list(tf.layers),
        "embed_tokens": tf.embed_tokens,
        "pos_embed": lambda ids: None,
        "final_ln": tf.norm,
        "lm_head": model.lm_head,
        "arch": "qwen2",
    }

################################################################################
# MAIN LOADER (dispatch)
################################################################################

def load_model_with_adapter(model_name: str):
    """
    Detect the architecture and return a unified interface for ANEE.
    This allows ANEE to run on GPT-2, Neo, J, LLaMA, Mistral, Falcon, Phi-2, Qwen-2.
    """

    config = AutoConfig.from_pretrained(model_name)
    arch = config.model_type.lower()

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if arch == "gpt2":
        return _adapt_gpt2(model)

    if arch == "gpt_neo":
        return _adapt_gptneo(model)

    if arch == "gptj":
        return _adapt_gptj(model)

    if arch == "llama":
        return _adapt_llama(model)

    if arch in ["falcon", "rwkv"]:  # falcon models show model_type='falcon'
        return _adapt_falcon(model)

    if arch == "mistral":
        return _adapt_mistral(model)

    if arch == "phi":
        return _adapt_phi2(model)

    if arch == "qwen2":
        return _adapt_qwen2(model)

    raise ValueError(f"[ANEE] Unsupported architecture: {arch}")
