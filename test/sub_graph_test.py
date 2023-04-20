import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../layer'))

from sub_graph import SubGraph
import torch

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # feature_num must equal hidden_size
    obj_num = 34 # objs of all batch
    vec_len = 30
    feature_num = 64
    hidden_size = 64
    input = torch.randn(obj_num, vec_len, feature_num, device = device)
    mask = torch.randn(obj_num, vec_len, feature_num // 2, device = device)
    model = SubGraph(hidden_size=hidden_size)
    model.to(device)

    out = model(input, mask)
    print(out.shape)
