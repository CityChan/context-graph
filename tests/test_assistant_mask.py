import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path("verl/utils/dataset/assistant_mask.py")
SPEC = importlib.util.spec_from_file_location("contextgraph_assistant_mask", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_qwen_style_template_gets_one_idempotent_generation_block():
    template = (
        "{%- for message in messages %}"
        "{%- if message.role == 'user' %}{{- message.content }}"
        "{%- elif message.role == \"assistant\" %}{{- message.content }}<|im_end|>"
        "{%- elif message.role == \"tool\" %}{{- message.content }}"
        "{%- endif %}{%- endfor %}"
    )

    patched = MODULE.chat_template_with_generation_blocks(template)

    assert patched.count("{%- generation %}") == 1
    assert patched.count("{%- endgeneration %}") == 1
    assert patched.replace("{%- generation %}", "").replace("{%- endgeneration %}", "") == template
    assert patched.index("{%- generation %}") < patched.index("{{- message.content }}<|im_end|>")
    assert patched.index("{%- endgeneration %}") < patched.index('{%- elif message.role == "tool" %}')
    assert MODULE.chat_template_with_generation_blocks(patched) == patched


class _CharacterTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        assert not tokenize
        rendered = "".join(
            f"<{message['role']}>{message.get('content', '')}</{message['role']}>"
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    ):
        import torch

        assert not add_special_tokens
        return {
            "input_ids": torch.tensor([[ord(char) for char in text]]),
            "attention_mask": torch.ones((1, len(text)), dtype=torch.long),
            "offset_mapping": torch.tensor([[[index, index + 1] for index in range(len(text))]]),
        }


def test_offset_fallback_masks_only_assistant_content_in_complete_render():
    pytest.importorskip("torch")
    tokenizer = _CharacterTokenizer()
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "repeat"},
        {"role": "assistant", "content": "repeat"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "answer"},
    ]
    rendered = tokenizer.apply_chat_template(messages)
    encoded = tokenizer(rendered)
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]

    loss_mask = MODULE.assistant_token_mask_from_offsets(
        tokenizer=tokenizer,
        processor=tokenizer,
        messages=messages,
        tools=None,
        apply_chat_template_kwargs={},
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    masked_text = "".join(chr(token) for token, mask in zip(input_ids.tolist(), loss_mask.tolist()) if mask)
    assert masked_text == "repeatanswer"
    assert int(loss_mask.sum()) == len("repeatanswer")
