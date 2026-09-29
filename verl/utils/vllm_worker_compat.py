"""Small, CPU-testable adapters for vLLM worker API changes."""

import inspect

import cloudpickle


def worker_wrapper_kwargs(wrapper_cls, vllm_config):
    """Each VERL RPC carries one kwargs entry, hence wrapper rpc/global rank 0.

    The actual distributed worker rank remains in init_worker's kwargs. Modern
    wrappers receive their config there instead of in the constructor.
    """
    signature = inspect.signature(wrapper_cls)
    kwargs = {}
    if "vllm_config" in signature.parameters:
        kwargs["vllm_config"] = vllm_config
    if "rpc_rank" in signature.parameters:
        kwargs["rpc_rank"] = 0
    if "global_rank" in signature.parameters:
        kwargs["global_rank"] = 0
    signature.bind(**kwargs)  # Fail early on an unknown required constructor arg.
    return kwargs


def execute_worker_method(wrapper, method, *args, **kwargs):
    legacy = getattr(type(wrapper), "execute_method", None)
    if legacy is not None:
        return legacy(wrapper, method, *args, **kwargs)
    if isinstance(method, str):
        # Preserve wrapper-owned init_device / cache methods before delegation.
        return getattr(wrapper, method)(*args, **kwargs)
    if isinstance(method, bytes):
        method = cloudpickle.loads(method)
    if callable(method):
        return method(wrapper, *args, **kwargs)
    raise TypeError(f"Unsupported worker RPC method type: {type(method).__name__}")
