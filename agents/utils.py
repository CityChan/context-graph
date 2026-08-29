import os
import time
import copy
import uuid
from collections.abc import Mapping
from unittest.mock import patch
from itertools import groupby
import re, unicodedata
from numbers import Integral
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
import aiohttp
import torch
from pydantic import BaseModel
from typing import Any, Optional
import asyncio, httpx
from envs.local_search import LocalSearch
from envs.alfworld_env import ALFWorldEnv
from .structured_outputs import normalize_structured_outputs


def select_env(ability, config, extra_info=None):
    # Select env
    if 'ALFWorld' in ability:
        EnvClass = ALFWorldEnv
    elif 'ScienceWorld' in ability:
        from envs.scienceworld_env import ScienceWorldEnv
        EnvClass = ScienceWorldEnv
    elif 'AppWorld' in ability:
        from envs.appworld_env import AppWorldEnv
        EnvClass = AppWorldEnv
    elif 'LocalSearch' in ability or 'GAIA' in ability:
        EnvClass = LocalSearch
    elif 'ScienceAgentBench' in ability:
        # Imported lazily so we don't pay the cost on BC-Plus / ALFWorld runs.
        from envs.scienceagent_env import ScienceAgentEnv
        EnvClass = ScienceAgentEnv
    elif 'DiscoveryBench' in ability:
        from envs.discoverybench_env import DiscoveryBenchEnv
        EnvClass = DiscoveryBenchEnv
    else:
        raise ValueError(
            f"Unknown ability: {ability}. Supported: ALFWorld@*, LocalSearch, GAIA, "
            "ScienceWorld@*, AppWorld@train, ScienceAgentBench, DiscoveryBench."
        )
    return EnvClass


async def call_openai(messages, model='gpt-5-nano', max_retries=3):
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    openai_url = os.getenv("OPENAI_URL")

    # If no proxy URL is set, fall back to the OpenAI SDK directly using OPENAI_API_KEY.
    if not openai_url:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "dummy":
            return "Error: no OPENAI_URL and no valid OPENAI_API_KEY"
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(model=model, messages=messages)
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[CALL OPENAI] Error after {max_retries} attempts: {str(e)}")
                    return f"Error after {max_retries} attempts: {str(e)}"
                await asyncio.sleep(1 * (attempt + 1))
        return ""

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=300.0) as c:
                r = await c.post(openai_url, json={
                    "model": model,
                    "messages": messages
                })
                r.raise_for_status()
                return r.json()["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[CALL OPENAI] Error after {max_retries} attempts: {str(e)}")
                return f"Error after {max_retries} attempts: {str(e)}"
            await asyncio.sleep(1 * (attempt + 1))
    return ""

def decode_conversation(input_ids: list[int], tokenizer) -> tuple[list[dict[str, str]], str]:
    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)
    pattern = re.compile(
        re.escape(tokenizer.bos_token)
        + r'(system|user|assistant|tool)\n'
        + r'(.*?)'
        + r'(?=' + re.escape(tokenizer.eos_token) + r')',
        re.DOTALL,
    )
    matches = pattern.findall(decoded_str)
    conversation = [{'role': role, 'content': content} for role, content in matches]
    return conversation, decoded_str

def truncate_text(
        text: str,
        max_lines: int | None = None,
        max_length: int | None = None,
        merge_repeat: bool = False,
        merge_num: int = 128,
        keep_tail_lines: int = 5,
) -> str:
    lines = text.splitlines()

    # 1) Merge repeated lines if requested
    if merge_repeat:
        merged: list[str] = []
        for line, group in groupby(lines):
            grp = list(group)
            cnt = len(grp)
            if cnt > merge_num:
                merged += [line] * 2
                merged.append(f"[This line repeated {cnt - 4} more times]")
                merged += [line] * 2
            else:
                merged += grp
        lines = merged

    # 2) Line-count truncation (keep last keep_tail_lines)
    if max_lines is not None and len(lines) > max_lines:
        total = len(lines)
        if max_lines <= keep_tail_lines + 1:
            lines = lines[:max_lines]
        else:
            head_count = max_lines - keep_tail_lines - 1
            head = lines[:head_count]
            tail = lines[-keep_tail_lines:]
            omitted = total - head_count - keep_tail_lines
            lines = head + [f"… {omitted} lines omitted …"] + tail

    # 3) Per-line character-length truncation
    if max_length is not None:
        truncated_lines: list[str] = []
        for line in lines:
            if len(line) > max_length:
                truncated_lines.append(line[:max_length] + "… (truncated)")
            else:
                truncated_lines.append(line)
        lines = truncated_lines
    return "\n".join(lines)


def is_weird(text, repeat_n=128, cjk_limit=128):
    s = unicodedata.normalize('NFKC', text)
    if re.search(rf'(.)\1{{{repeat_n - 1},}}|(.{{2,12}})\2{{{repeat_n - 1},}}', s):
        return True
    CJK = ((0x4E00, 0x9FFF), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
    cjk_count = sum(any(a <= ord(c) <= b for a, b in CJK) for c in s)
    return cjk_count >= cjk_limit or (len(s) > 0 and cjk_count / len(s) > 0.8)

class LLMClass:
    async def create_completion(self, input_ids, **kwargs):
        raise NotImplemented


def _normalize_token_ids(token_ids, *, source="tokenizer") -> list[int]:
    """Convert tokenizer outputs to the flat integer IDs expected by vLLM.

    Transformers' tokenizers backend may return ``tokenizers.Encoding``
    objects (or lists of them) where older backends returned ``list[int]``.
    Keep that backend difference at the tokenizer boundary instead of letting
    non-integer objects reach vLLM's ``TokensPrompt`` validation.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()

    if isinstance(token_ids, Mapping):
        if "input_ids" not in token_ids:
            raise TypeError(
                f"{source} returned {type(token_ids).__name__} without input_ids"
            )
        token_ids = token_ids["input_ids"]

    encoding_ids = getattr(token_ids, "ids", None)
    if encoding_ids is not None and not isinstance(token_ids, (list, tuple)):
        token_ids = encoding_ids

    if isinstance(token_ids, Integral) and not isinstance(token_ids, bool):
        return [int(token_ids)]
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError(
            f"{source} returned unsupported token IDs of type "
            f"{type(token_ids).__name__}; expected integers or Encoding.ids"
        )

    normalized = []
    for item in token_ids:
        if isinstance(item, Integral) and not isinstance(item, bool):
            normalized.append(int(item))
        else:
            normalized.extend(_normalize_token_ids(item, source=source))
    return normalized

class CallLLM(LLMClass):  # Call LLM in Verl RL env
    def __init__(
        self,
        url,
        tokenizer,
        config,
        loop,
        sampling_params=None,
        **kwargs,
    ):
        self.server_manager = url
        self.tokenizer = tokenizer
        self.config = config
        self.loop = loop
        self.call_openai = getattr(config.plugin, "call_openai", None)
        # AgentLoopWorker already resolves train-vs-validation sampling here
        # (notably validation temperature=0 and calculate_log_probs). Keep an
        # immutable copy so concurrent turns cannot mutate the shared dict.
        self.sampling_params = dict(sampling_params or {})

    async def _create_completion(self, input_ids, **kwargs):
        from uuid import uuid4

        input_ids = _normalize_token_ids(input_ids, source="agent prompt")
        structured_outputs = kwargs.pop('structured_outputs', None)

        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_len = min(max_len, self.config.prompt_length + self.config.response_length)
        max_new_tokens = max_len - len(input_ids)
        # Per-turn cap on generated tokens — without this, the agent fills
        # the full response_length budget in a single turn (esp. at val where
        # do_sample=False produces long deterministic thinking), and there's
        # no budget left for subsequent turns + the final <function=finish>.
        # NOTE: this used to assign to a local `max_tokens` variable that was
        # never read (line 176 below uses `max_new_tokens`), so the cap was
        # silently ignored — fix surfaced when val/overlong_rate hit 0.96.
        if hasattr(self.config, 'plugin') and getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_new_tokens = min(max_new_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_new_tokens = min(max_new_tokens, kwargs['max_new_tokens'])

        if max_new_tokens < 10:
            print(f"[DEBUG] max_new_tokens {max_new_tokens}, skip rollout")
            return None

        uid = kwargs.pop('uid', None) or uuid4().hex

        sampling_params = dict(self.sampling_params)
        sampling_params.update(kwargs.pop('sampling_params', None) or {})
        sampling_params.setdefault('temperature', 1.0)
        sampling_params.setdefault('top_p', 1.0)
        sampling_params['max_tokens'] = min(
            int(sampling_params.get('max_tokens', max_new_tokens)),
            max_new_tokens,
        )
        if structured_outputs is not None:
            if sampling_params.get('guided_decoding') is not None:
                raise ValueError(
                    "structured_outputs and guided_decoding cannot both be set"
                )
            sampling_params['structured_outputs'] = (
                normalize_structured_outputs(structured_outputs)
            )

        output = await self.server_manager.generate(
            request_id=uid,
            prompt_ids=input_ids,
            sampling_params=sampling_params,
            image_data=None,
        )

        if output is None or len(output.token_ids) == 0:
            return None

        if sampling_params.get('logprobs', False):
            if getattr(output, 'log_probs', None) is None:
                raise RuntimeError(
                    "Rollout log-probs were requested but the rollout server "
                    "returned none. Refusing to replace them with zeros."
                )
            if len(output.log_probs) != len(output.token_ids):
                raise RuntimeError(
                    "Rollout log-prob/token length mismatch: "
                    f"{len(output.log_probs)} != {len(output.token_ids)}"
                )

        response_text = await self.loop.run_in_executor( None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True))

        return {
            "choices": [{
                "message": {
                    "content": response_text,
                    "raw_output_ids": output.token_ids,
                    "response_log_probs": (
                        output.log_probs
                        if getattr(output, 'log_probs', None) is not None
                        else [0.0] * len(output.token_ids)
                    ),
                    "extra_data": {"input_ids": input_ids},
                    "metrics": {}
                }
            }]
        }

    async def create_completion(self, input_ids, **kwargs):
        completion = await self._create_completion(input_ids, **kwargs)
        return completion


class CallAPI(LLMClass):  # Call external API (OpenAI)
    def __init__(self, url, tokenizer, config, **kwargs):
        self.tokenizer = tokenizer
        self.config = config
        self.model = url
        from openai import AsyncOpenAI
        import os
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", None)  # Optional custom base URL
        )

    async def close(self):
        """Close the underlying async HTTP transport before its event loop exits."""
        await self.client.close()

    async def create_completion(self, input_ids, **kwargs):
        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_tokens = min(max_len, self.config.prompt_length + self.config.response_length) - len(input_ids)

        if getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_tokens = min(max_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_tokens = min(max_tokens, kwargs.pop('max_new_tokens'))

        if max_tokens < 10:
            return None
        messages = kwargs.get('messages', None)
        if messages is None:
            messages = decode_conversation(input_ids, self.tokenizer)[0]
        structured_outputs = kwargs.pop('structured_outputs', None)

        for attempt in range(5):
            try:
                request = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
                temperature = getattr(self.config.plugin, "temperature", None)
                top_p = getattr(self.config.plugin, "top_p", None)
                reasoning_effort = getattr(self.config.plugin, "reasoning_effort", None)
                if temperature is not None:
                    request["temperature"] = float(temperature)
                if top_p is not None:
                    request["top_p"] = float(top_p)
                extra_body = {}
                if reasoning_effort:
                    # DeepSeek-V4 exposes reasoning controls through its
                    # custom chat-template kwargs. Keep this in extra_body so
                    # older OpenAI SDKs can forward it to a local vLLM server.
                    if reasoning_effort == "non-thinking":
                        chat_template_kwargs = {"thinking": False}
                    else:
                        chat_template_kwargs = {
                            "thinking": True,
                            "reasoning_effort": str(reasoning_effort),
                        }
                    extra_body["chat_template_kwargs"] = chat_template_kwargs
                if structured_outputs is not None:
                    extra_body["structured_outputs"] = structured_outputs
                if extra_body:
                    request["extra_body"] = extra_body
                response = await self.client.chat.completions.create(**request)

                text = response.choices[0].message.content or ""
                text_ids = _normalize_token_ids(
                    self.tokenizer.encode(text, add_special_tokens=False),
                    source="tokenizer.encode",
                )

                return {
                    "choices": [{
                        "message": {
                            "content": text,
                            "raw_output_ids": text_ids,
                            "response_log_probs": [0.0] * len(text_ids),
                            "extra_data": {"input_ids": input_ids},
                            "metrics": {"usage": {
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else len(text_ids),
                                "total_tokens": response.usage.total_tokens if response.usage else 0,
                            }}
                        }
                    }]
                }
            except Exception as e:
                if attempt == 4:
                    print(f"[CallAPI ERROR] Failed after 5 attempts: {e}")
                    return None
                wait_time = 2 ** attempt
                print(f"[CallAPI] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        return None


def _chat_template_kwargs(config) -> dict:
    """Return tokenizer.apply_chat_template kwargs from either config shape."""
    data_cfg = getattr(config, "data", None)
    if data_cfg is None and hasattr(config, "actor_rollout_ref"):
        data_cfg = getattr(config, "data", None)
    kwargs = {}
    if data_cfg is not None:
        raw = getattr(data_cfg, "apply_chat_template_kwargs", None)
        if raw:
            kwargs = dict(raw)
    plugin_cfg = getattr(config, "plugin", None)
    if plugin_cfg is not None:
        raw = getattr(plugin_cfg, "apply_chat_template_kwargs", None)
        if raw:
            kwargs.update(dict(raw))
        if hasattr(plugin_cfg, "qwen_enable_thinking"):
            raw_thinking = getattr(plugin_cfg, "qwen_enable_thinking")
            if isinstance(raw_thinking, str):
                kwargs["enable_thinking"] = raw_thinking.lower() in ("1", "true", "yes")
            else:
                kwargs["enable_thinking"] = bool(raw_thinking)
    env_thinking = os.environ.get("QWEN_ENABLE_THINKING")
    if env_thinking is not None:
        kwargs["enable_thinking"] = env_thinking.lower() in ("1", "true", "yes")
    return kwargs


def _apply_chat_template(tokenizer, chat, config, **kwargs):
    template_kwargs = _chat_template_kwargs(config)
    template_kwargs.update(kwargs)
    rendered = tokenizer.apply_chat_template(chat, **template_kwargs)
    if template_kwargs.get("tokenize", True):
        return _normalize_token_ids(rendered, source="tokenizer.apply_chat_template")
    return rendered


def truncate_prompt(chat, prompt_length, tokenizer, prompt_turn, config=None):
    exceed_len = len(_apply_chat_template(tokenizer, chat[:prompt_turn], config)) + 8 - prompt_length
    _cut_idx = 0
    while exceed_len > 0:  # truncate long user prompt
        print('[PROMPT] now exceed', exceed_len, 'work on cut turn', _cut_idx)
        content_ids = _normalize_token_ids(
            tokenizer.encode(chat[_cut_idx]['content'], add_special_tokens=False),
            source="tokenizer.encode",
        )
        chat[_cut_idx]['content'] = tokenizer.decode(
            content_ids[exceed_len + 4:], add_special_tokens=False)
        exceed_len = len(_apply_chat_template(tokenizer, chat[:prompt_turn], config)) + 8 - prompt_length
        _cut_idx = _cut_idx + 1
        if _cut_idx >= prompt_turn:
            break
    return chat


class AgentContext:
    # Manage context of an agent
    def __init__(self, chat, tokenizer, config, prompt_turn=2):
        self.tokenizer = tokenizer
        self.config = config
        self.init_len = len(chat)
        self.prompt_turn = prompt_turn

        # Support both inference and training config styles
        if hasattr(config, 'actor_rollout_ref'):
            # Training config (VERL)
            self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
            self.response_length = config.actor_rollout_ref.rollout.response_length
        else:
            # Inference config
            self.prompt_length = config.prompt_length
            self.response_length = config.response_length

        self.context_uid = str(uuid.uuid4())

        self.chat = copy.deepcopy([turn for turn in chat])
        self.chat = truncate_prompt(self.chat, config.prompt_length, tokenizer, prompt_turn, config)
        self.chat_completions = [None for _ in range(len(self.chat))]
        self.chat_ids = [self.get_turn_context(i) for i in range(len(self.chat))]
        self.log_probs = [[0.0] * len(turn) for turn in self.chat_ids]
        self.token_mask = [[False] * len(turn) for turn in self.chat_ids]
        self.additional_info = [None for _ in self.chat_ids]
        self.generation_prompt = None
        self.metrics = None
        self.prompt_ids_len = len(sum(self.chat_ids[:prompt_turn], []))

    def _render_prefix(self, chat):
        """Render a complete-enough chat prefix for incremental accounting.

        Some modern templates (including Qwen3.6) reject a system-only
        prefix with ``No user query found in messages`` even though the full
        system+user conversation is valid. Defer those system tokens to the
        first renderable prefix; the combined prompt token count remains
        exact and later assistant/user turns retain normal segmentation.
        """
        if not chat:
            return []
        try:
            return _apply_chat_template(
                self.tokenizer, chat, self.config,
                add_generation_prompt=False, tokenize=True
            )
        except Exception:
            if not any(turn.get('role') == 'user' for turn in chat):
                return []
            raise

    def get_turn_context(self, i):
        tokens = self._render_prefix(self.chat[:i + 1])
        prev = self._render_prefix(self.chat[:i]) if i > 0 else []
        turn_tokens = tokens[len(prev):]
        return turn_tokens

    def get_generation_prompt(self):
        if self.generation_prompt is None:
            tokens = _apply_chat_template(
                self.tokenizer, self.chat, self.config,
                add_generation_prompt=False, tokenize=True
            )
            add_tokens = _apply_chat_template(
                self.tokenizer, self.chat, self.config,
                add_generation_prompt=True, tokenize=True
            )
            self.generation_prompt = add_tokens[len(tokens):]
        return self.generation_prompt

    def messages(self):
        return self.chat

    def context_ids(self, messages=None):
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def context(self, turn_cut: int=None):
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def append(self, turn, completion=None, additional_info=None):
        self.chat.append(turn)
        self.chat_completions.append(completion)
        self.additional_info.append(additional_info)
        if completion is None:
            self.chat_ids.append(self.get_turn_context(len(self.chat) - 1))
            self.log_probs.append([0.0] * len(self.chat_ids[-1]))
            self.token_mask.append([False] * len(self.chat_ids[-1]))
        else:
            completion_tokens = completion["choices"][0]["message"]["raw_output_ids"]
            completion_log_probs = completion["choices"][0]["message"]["response_log_probs"]
            self.chat_ids.append(self.get_generation_prompt() + completion_tokens)
            self.log_probs.append([0.0] * len(self.get_generation_prompt()) + completion_log_probs)
            self.token_mask.append([False] * len(self.get_generation_prompt()) + [True] * len(completion_tokens))
            if len(completion_tokens) == 0 or completion_tokens[-1] != self.tokenizer.eos_token_id:
                self.chat_ids[-1].append(self.tokenizer.eos_token_id)
                self.log_probs[-1].append(0.0)
                self.token_mask[-1].append(False)

    def rollback(self, k=1):
        self.chat = self.chat[:-k]
        self.chat_completions = self.chat_completions[:-k]
        self.chat_ids = self.chat_ids[:-k]
        self.log_probs = self.log_probs[:-k]
        self.token_mask = self.token_mask[:-k]
        self.additional_info = self.additional_info[:-k]

    def replace_user_turn(self, turn_idx, new_content):
        """Replace a non-assistant turn's content with shorter text (sliding-window
        compression for old observations). Assistant turns must not be touched —
        their token_mask carries gradient mass.
        """
        assert self.chat[turn_idx]['role'] != 'assistant', \
            f"Cannot replace assistant turn at {turn_idx}"
        self.chat[turn_idx]['content'] = new_content
        # Concatenative chat templates: only this turn's tokens depend on its
        # own content; subsequent turns' chat_ids are unaffected.
        self.chat_ids[turn_idx] = self.get_turn_context(turn_idx)
        self.log_probs[turn_idx] = [0.0] * len(self.chat_ids[turn_idx])
        self.token_mask[turn_idx] = [False] * len(self.chat_ids[turn_idx])

    def replace_assistant_turn(self, turn_idx, new_content, *, audit_info=None):
        """Replace a controller-canonicalized assistant turn safely.

        Generated token IDs and log probabilities describe the raw response,
        not the repaired representation. Retokenize the canonical turn and
        mask it from online policy-gradient updates; API-generated SFT is built
        later from the canonical ``messages`` field.
        """
        assert self.chat[turn_idx]['role'] == 'assistant', \
            f"Expected assistant turn at {turn_idx}"
        self.chat[turn_idx]['content'] = new_content
        self.chat_ids[turn_idx] = self.get_turn_context(turn_idx)
        self.log_probs[turn_idx] = [0.0] * len(self.chat_ids[turn_idx])
        self.token_mask[turn_idx] = [False] * len(self.chat_ids[turn_idx])
        if audit_info is not None:
            existing = self.additional_info[turn_idx]
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(audit_info)
            self.additional_info[turn_idx] = merged

    def get_metrics(self):
        if self.metrics is None:
            return {}
        return self.metrics

    async def get_data(self):
        prompt_turn = self.prompt_turn
        prompt_length = self.prompt_length
        response_length = self.response_length

        prompt_ids = sum(self.chat_ids[:prompt_turn], [])
        if len(prompt_ids) > prompt_length:
            print('[PROMPT] prompt truncated')
            prompt_ids = prompt_ids[-prompt_length:]

        response_ids = sum(self.chat_ids[prompt_turn:], [])[:response_length]
        response_logprobs = sum(self.log_probs[prompt_turn:], [])[:response_length]
        response_mask = [1 if m else 0 for turn in self.token_mask[self.prompt_turn:] for m in turn][:response_length]
        process_reward_mask = sum([[info.get('process_reward', 0) if isinstance(info, dict) else 0] * len(turn)
                                   for turn, info in zip(self.chat_ids, self.additional_info)][prompt_turn:], [])
        process_reward_mask = [p * m for p, m in zip(process_reward_mask, response_mask)][:response_length]
        return {
            'prompt_ids': prompt_ids,
            'response_ids': response_ids,
            'response_logprobs': response_logprobs,
            'response_mask': response_mask,
            'process_reward_mask': process_reward_mask,
            'num_turns': len(self.chat_ids),
            'messages': self.chat,
        }


class Agent(AgentContext):
    # Agent utils
    def __init__(self, llm_client, conversations, tokenizer, config, prompt_turn=2):
        super().__init__(conversations, tokenizer, config, prompt_turn=prompt_turn)
        self.llm_client = llm_client
        self.retry_cjk = getattr(config.plugin, "retry_cjk", 0)
        self.info_cache = {}
        self.tool_format_repairs = []

    async def step(self, max_new_tokens=None, retry_cjk=0, completion_kwargs=None):
        prompt = self.context()
        max_len = self.prompt_ids_len + self.config.response_length
        if max_new_tokens is not None:
            max_len = min(len(prompt) + max_new_tokens, 131072)
        completion_kwargs = dict(completion_kwargs or {})
        completion = await self.llm_client.create_completion(
            prompt,
            uid=self.context_uid,
            max_len=max_len,
            messages=self.chat,
            **completion_kwargs,
        )
        if completion is None:
            return None
        response = completion["choices"][0]["message"]["content"]
        self.append({'role': 'assistant', 'content': response}, completion)
        if getattr(self.config.plugin, "controller_owned_tool_formatting", False):
            from .tool_protocol import canonicalize_tool_call_text

            canonical, repairs = canonicalize_tool_call_text(response)
            if repairs:
                raw_response = response
                response = canonical
                repair_record = {
                    "turn_idx": len(self.chat) - 1,
                    "raw_response": raw_response,
                    "canonical_response": canonical,
                    "repairs": repairs,
                }
                self.tool_format_repairs.append(repair_record)
                self.replace_assistant_turn(
                    len(self.chat) - 1,
                    canonical,
                    audit_info={
                        "tool_format_repaired": True,
                        "raw_assistant_content": raw_response,
                        "tool_format_repairs": repairs,
                    },
                )
        return response

    async def react(self, run_action, max_turn=64, max_tokens=None, session_timeout=60 * 60,
                    should_continue=None, summary_prompt=None, safe_finish=None, observation_prompt=None):
        # Run react for max_turn turn
        if should_continue is None:
            should_continue = lambda st: True
        session_start_time = time.time()
        iteration = 0
        if max_tokens is not None:
            max_tokens = max_tokens - 512
        else:
            max_tokens = self.config.response_length - 512

        last_response = None
        response = None
        init_len = len(self.context(turn_cut=self.prompt_turn))
        while iteration < max_turn:
            if time.time() - session_start_time > session_timeout:  # TODO add session timeout
                print('[SESSION] Session Timeout')
                break
            if len(self.context()) - init_len > max_tokens:  # summary
                break

            iteration += 1
            response = await self.step()
            if response is None:
                break

            if not should_continue(response):
                last_response = response
                break
            if safe_finish is not None and safe_finish(response) is not None:
                observation = safe_finish(response)
            else:
                observation = await run_action(response)
            if observation is None:
                break
            if observation_prompt:
                observation += '\n' + observation_prompt
            self.append({'role': 'user', 'content': observation, })

        if last_response is None and summary_prompt is not None:
            if len(self.context()) - init_len > self.config.response_length - 1024:  # summary
                self.rollback(k=2)
            if self.chat[-1]['role'] == 'user':
                self.append({'role': 'assistant', 'content': "", })
            self.append({'role': 'user', 'content': summary_prompt, })
            last_response = await self.step(max_new_tokens=4096)
        elif last_response is None:
            last_response = str(response)

        return {'last_response': last_response, 'iteration': iteration}

    def set_process_reward(self, turn, reward):
        if isinstance(turn, str) and turn.lower() == 'all':
            turn = [i for i in range(len(self.chat))]
        if not isinstance(turn, list):
            turn = [turn]
        for i in turn:
            if i <= 0:
                continue
            if i > len(self.chat) - 1:
                continue
            if self.chat_completions[i] is None:
                continue
            if self.additional_info[i] is None:
                self.additional_info[i] = {}
            self.additional_info[i]['process_reward'] = reward

    def set_cache(self, key, value):
        self.info_cache[key] = value


@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    is_train: bool
    tokenizer: PreTrainedTokenizer | AutoTokenizer | None = None
    llm_client: LLMClass = None

class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics.

    Kept local to avoid importing `verl.experimental.agent_loop`, whose package
    initializer registers script entry points and can circular-import agents.
    """

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output compatible with verl's post-processing fields."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: Optional[list[float]] = None
    routed_experts: Optional[Any] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    reward_score: Optional[float] = None
    num_turns: int = 0
    metrics: AgentLoopMetrics
    extra_fields: dict[str, Any] = {}


async def run_action(env, response):
    try:
        try:
            act = time.time()
            env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)
            if time.time() - act > 10:
                print('Action Cost', time.time() - act)
        except asyncio.TimeoutError:
            print('[ACTION] Action timed out after 120 seconds')
            env_return = {'observation': 'Action timed out after 120 seconds'}
        if 'action' in env_return:
            action, arguments = env_return['action'], env_return.get('arguments', {})
            if action == 'finish':
                return None
        elif env_return.get('observation', None) == 'finish':
            return None
        observation = env_return.pop('observation', 'Empty')
    except Exception as e:
        observation = f"Error: {e}"
    return observation
