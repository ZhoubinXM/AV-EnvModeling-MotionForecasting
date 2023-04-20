import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../layer'))

from deepfm import DeepFM
import torch

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cate_fea_nuniqs = [3, 5, 8]
    cate_features = [[2, 3, 1], [1, 3, 2]]
    dense_features = [[0.1, 3.2, 1.3, 2.1], [1.2, 3.1, 2.1, 3.2]]
    model = DeepFM(cate_fea_nuniqs, nume_fea_size=4, emb_size=16)
    model.to(device)

    cate_features = torch.tensor(cate_features, device=device)
    dense_features = torch.tensor(dense_features, device=device)

    out = model(cate_features, dense_features)
    print(out.shape)
    # print(out)
