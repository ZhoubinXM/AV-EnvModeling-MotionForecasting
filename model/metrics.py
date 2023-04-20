import torch


def get_metrics(forecasted_trajectory, gt_trajectory, miss_threshold):
    dx = forecasted_trajectory[:, :, :, 0] - gt_trajectory[:, :, :, 0]
    dy = forecasted_trajectory[:, :, :, 1] - gt_trajectory[:, :, :, 1]
    dis = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
    min_fde, indices = torch.min(dis[:, :, -1], 1)
    fde = torch.mean(min_fde)
    gather_indices = indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dis.shape[2])
    min_dis = torch.gather(dis, 1, gather_indices)
    ade = torch.mean(torch.mean(min_dis))
    miss_rate = torch.sum(min_fde > miss_threshold) / dis.shape[0]
    return {"minADE": ade, "minFDE": fde, "MR": miss_rate}


# def get_ade(forecasted_trajectory, gt_trajectory):
#     dx = forecasted_trajectory[:, :, :, 0] - gt_trajectory[:, :, :, 0]
#     dy = forecasted_trajectory[:, :, :, 1] - gt_trajectory[:, :, :, 1]
#     dis = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
#     val, id = torch.min(dis[:, :, -1], 1)
#     ade = torch.mean(torch.mean(dis))
#     return ade

# def get_fde(forecasted_trajectory, gt_trajectory):
#     dx = forecasted_trajectory[:, :, -1, 0] - gt_trajectory[:, :, -1, 0]
#     dy = forecasted_trajectory[:, :, -1, 1] - gt_trajectory[:, :, -1, 1]
#     dis = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
#     fde = torch.mean(dis)
#     return fde

# def get_miss_rate(forecasted_trajectory, gt_trajectory, miss_threshold):
#     dx = forecasted_trajectory[:, :, -1, 0] - gt_trajectory[:, :, -1, 0]
#     dy = forecasted_trajectory[:, :, -1, 1] - gt_trajectory[:, :, -1, 1]
#     dis = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
#     miss_rate = torch.sum(dis > miss_threshold) / dis.shape[0]
#     return miss_rate


def cal_metrics(forecasted_trajectory, gt_trajectory, miss_threshold):
    batch, K, _ = forecasted_trajectory.size()
    forecasted_trajectory = forecasted_trajectory.reshape(batch, K, 30, 2)
    gt_trajectory = gt_trajectory.reshape(batch, 1, 30, 2)
    metric_results = get_metrics(forecasted_trajectory, gt_trajectory, miss_threshold)
    return metric_results
