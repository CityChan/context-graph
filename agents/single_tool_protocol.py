"""Bounded format recovery for evaluation adapters that dispatch one tool."""

MAX_FORMAT_ERRORS = 3
PROTOCOL = {"tool_prompt_profile": "search_single_v1", "format_retry_limit": MAX_FORMAT_ERRORS,
            "format_retry_context": "last rejected response, 512-token tail plus truncation marker"}


def retry_context(text, error, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=False)
    rejected = text if len(ids) <= 512 else (
        "[Earlier rejected output omitted]\n" + tokenizer.decode(ids[-512:], skip_special_tokens=True))
    correction = (f"Format correction: {error}. Your previous response was rejected; no tool was executed. "
                  "Return exactly one complete function call and wait for its result. "
                  "Do not batch calls. Keep any required compression block before that one call.")
    return rejected, correction
