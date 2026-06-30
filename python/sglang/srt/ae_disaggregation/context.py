"""Janus-compatible mutable runtime context for AE workers."""

# Kept as a dictionary to preserve the Janus communication/runner control flow.
global_server_args_dict = {}
