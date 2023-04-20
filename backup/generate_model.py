from inspect import trace
import torch
import train_eval.utils as u
import numpy as np
import onnxruntime as ort
import onnx

# device = torch.device("cuda")

input = {
    'veh_cate_features': np.array([[2, 3, 1, 1]]),
    'veh_dense_features': np.array([[0.1, 3.2, 1.3, 2.1]]),
    'driver_cate_features': np.array([[2, 3, 1, 2]]),
    'driver_dense_features': np.array([[0.1, 3.2, 1.3, 2.1, 1.3, 0.1, 3.2, 1.3, 2.1]]),
    'polylines': np.random.randn(256, 19, 128),
    'polynum': np.array([16]),
    'attention_mask': np.random.randn(256, 19, 128 // 2)
}

torch_input = {}

for key, value in input.items():
    torch_input[key] = torch.from_numpy(value)

torch_input = u.convert_double_to_float(torch_input)

for key, value in torch_input.items():
    print(key, " ", value.device)

# inputs = torch_input
# veh_cate_features = inputs['veh_cate_features']
# veh_dense_features = inputs['veh_dense_features']
# driver_cate_features = inputs['driver_cate_features']
# driver_dense_features = inputs['driver_dense_features']
# polylines = inputs['polylines']
# attention_mask = inputs['attention_mask']
# polynum = inputs['polynum']

model = torch.load("output/model/best_adms_model.pth")
model = model.to("cpu")
model.eval()
# out = model(veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
#             attention_mask)
# print(out)

(veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
 attention_mask) = model.preprocess_inputs(torch_input)

torch.save(
    (veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum, attention_mask),
    "output/model/data.pt")

traced_script_module = torch.jit.trace(
    model,
    (veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum, attention_mask))
traced_script_module.save("output/model/adms_model_script.pt")

jit_model = torch.jit.load("output/model/adms_model_script.pt")

print(jit_model)

torch.onnx.export(
    model,
    (veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum, attention_mask),
    r"output/model/adms_model.onnx",
    export_params=True,
    input_names=[
        'veh_cate_features', 'veh_dense_features', 'driver_cate_features', 'driver_dense_features', 'polylines', 'polynum',
        'attention_mask'
    ],
    #   output_names=['traj', 'prob'],
    output_names=['ttc'],
    verbose=False,
    opset_version=11)

model_onnx = onnx.load("output/model/adms_model.onnx")
onnx.checker.check_model(model_onnx)
ort_session = ort.InferenceSession("output/model/adms_model.onnx", providers=['CPUExecutionProvider'])

for input in ort_session.get_inputs():
    print(input.name, input.shape, input.type)
