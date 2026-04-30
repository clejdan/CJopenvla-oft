# Save this as 'check_install.py'
import torch
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
print("VLA Constants loaded successfully!")

try:
    import flash_attn
    print(f"Flash Attention version {flash_attn.__version__} is active.")
except ImportError:
    print("Flash Attention not found.")
