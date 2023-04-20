import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../layer'))

from global_graph import GlobalGraph
import torch

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # feature_num must equal hidden_size
    batch_size = 32
    max_poly_num = 30
    feature_num = 64
    hidden_size = 64
    input = torch.randn(batch_size, max_poly_num, feature_num, device = device)
    mask = torch.randn(batch_size, max_poly_num, max_poly_num, device = device)
    model = GlobalGraph(hidden_size=hidden_size)
    model.to(device)

    out = model(input, mask)
    print(out.shape)
