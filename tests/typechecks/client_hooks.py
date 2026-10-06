"""Pyright regression coverage for typed collections passed to the public API."""

from vllm_lens import Hook, LinearProbe
from vllm_lens.client import VLLMLensClient


def check_client_hook_collections(
    client: VLLMLensClient, hooks: list[Hook], probes: list[LinearProbe]
) -> None:
    client.register_hooks(hooks)
    client.generate("Hello", hooks=hooks)
    client.chat([{"role": "user", "content": "Hello"}], hooks=hooks)
    client.register_hooks(probes)
    client.generate("Hello", hooks=probes)
    client.chat([{"role": "user", "content": "Hello"}], hooks=probes)
