import os
import torch
import numpy as np
from maddpg_agents import WirelessActorNetwork

def export_to_cpp_header(model, filepath="include/weights.h"):
    state_dict = model.state_dict()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("// Auto-generated fixed-point weights configuration file\n")
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n#include <cstdint>\n\n")
        
        for key, tensor in state_dict.items():
            arr = tensor.cpu().numpy()
            q_arr = np.round(arr * 128.0).astype(np.int16)
            clean_name = key.replace('.', '_')
            
            if len(q_arr.shape) == 1:
                f.write(f"const int16_t {clean_name} [{q_arr.shape[0]}] = {{")
                f.write(", ".join(map(str, q_arr)))
                f.write("};\n\n")
            elif len(q_arr.shape) == 2:
                f.write(f"const int16_t {clean_name} [{q_arr.shape[0]}][{q_arr.shape[1]}] = {{\n")
                for row in q_arr:
                    f.write("    {" + ", ".join(map(str, row)) + "},\n")
                f.write("};\n\n")
        f.write("#endif // WEIGHTS_H\n")

if __name__ == "__main__":
    trained_model = WirelessActorNetwork(state_dim=128, action_dim=16)
    export_to_cpp_header(trained_model)
    print("[Success] PyTorch model layers successfully quantized to include/weights.h")
