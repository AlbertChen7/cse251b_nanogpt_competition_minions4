import torch
import os

from train_gpt_distill import GPT, GPTConfig
import __main__

def average_checkpoints(checkpoint_paths, output_path):
    print(f"averaging {len(checkpoint_paths)} checkpoints:")
    
    avg_state_dict = torch.load(checkpoint_paths[0], map_location='cpu')['model']
    
    # add
    for path in checkpoint_paths[1:]:
        state_dict = torch.load(path, map_location='cpu')['model']
        for key in avg_state_dict.keys():
            avg_state_dict[key] += state_dict[key]
    # average
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= len(checkpoint_paths)
        
    # save model
    torch.save({'model': avg_state_dict}, output_path)
    print(f"Averaged model saved to {output_path}")

if __name__ == "__main__":
    average_checkpoints(["log/model_15000.pt", "log/model_19072.pt"], "log/model_averaged.pt")