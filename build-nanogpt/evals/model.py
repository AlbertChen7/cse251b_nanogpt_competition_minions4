import torch

import sys
import os

# Adds the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_muon_gpt import GPT, GPTConfig
import __main__

class NanoGPTAdapter(torch.nn.Module):
    """
    A wrapper class to make Andrej's model play nicely with the eval script.
    It strips the loss from the output and slices the padded vocabulary.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # Andrej's forward pass returns (logits, loss). We only want logits.
        logits, _ = self.model(input_ids)
        
        # Andrej pads the vocab to 50304 for TensorCore efficiency. 
        # The eval script expects exactly 50257. We slice it here.
        return logits[:, :, :50257]

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load your trained model from a checkpoint.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # --- THE FIX ---
    # Trick PyTorch's pickle into finding GPTConfig in the current main script
    __main__.GPTConfig = GPTConfig
    # ---------------
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # ... (the rest of the code remains exactly the same) ...
    config = GPTConfig(
        block_size=1024,
        vocab_size=50304, 
        n_layer=4,        
        n_head=4,         
        n_embd=512        
    )

    # print(f"Loading checkpoint from {checkpoint_path}...")
    # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # # 1. Recreate the model configuration
    # # IMPORTANT: If you trained a small model (e.g., 4 layers), you MUST 
    # # update these numbers to match your training command!
    # config = GPTConfig(
    #     block_size=1024,
    #     vocab_size=50304, # Must be 50304 to match how it was trained
    #     n_layer=4,        # CHANGE THIS if you used default 12
    #     n_head=4,         # CHANGE THIS if you used default 12
    #     n_embd=512        # CHANGE THIS if you used default 768
    # )
    
    # 2. Initialize the raw architecture
    model = GPT(config)
    
    # 3. Clean up the state dictionary keys
    # (Removes the '_orig_mod.' prefix if the model was compiled)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # 4. Load the weights into the model
    model.load_state_dict(state_dict)
    
    # 5. Wrap the model in our adapter
    wrapped_model = NanoGPTAdapter(model)
    
    # 6. Send to GPU and set to evaluation (read-only) mode
    wrapped_model.to(device)
    wrapped_model.eval()
    
    return wrapped_model