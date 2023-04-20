import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../layer'))


from reserve.model import ADMS
import torch

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch = 2
    veh_cate_fea_nuniqs = [3, 5, 8]
    driver_cate_fea_nuniqs = [4, 7, 8, 6]
    veh_cate_features = [[2, 3, 1], [1, 3, 2]]
    driver_cate_features = [[2, 3, 1, 2], [1, 3, 2, 2]]
    veh_dense_features = [[0.1, 3.2, 1.3, 2.1], [1.2, 3.1, 2.1, 3.2]]
    driver_dense_features = [[0.1, 3.2, 1.3, 2.1, 1.3], [1.2, 3.1, 2.1, 3.2, 4.5]]
    # model
    feature_size = 128
    model = ADMS(veh_cate_fea_nuniqs=veh_cate_fea_nuniqs,
                 driver_cate_fea_nuniqs=driver_cate_fea_nuniqs,
                 veh_nume_fea_size=4,
                 driver_nume_fea_size=5,
                 feature_size = feature_size,
                 mode="train",
                 device=device)
    model.to(device)

    veh_cate_features = torch.tensor(veh_cate_features, device=device)
    driver_cate_features = torch.tensor(driver_cate_features, device=device)
    veh_dense_features = torch.tensor(veh_dense_features, device=device)
    driver_dense_features = torch.tensor(driver_dense_features, device=device)
    polylines = torch.randn(9, 19, 128, device=device)
    poly_num = torch.tensor([5,4], device=device)
    attention_mask = torch.randn(9, 19, 128 // 2, device = device)

    out = model(veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, poly_num,
                attention_mask)
    print(out)
