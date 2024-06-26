import onnxruntime
import numpy as np

def predict(input_img):
    model_onnx = '/content/drive/MyDrive/Brain-Tumor-Segmentation/checkpoints/brain-mri-unet.onnx'

    session = onnxruntime.InferenceSession(model_onnx, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_img})
    pred_mask = np.array(result).astype(np.float32)
    pred_mask = pred_mask * 255
    pred_mask = pred_mask[0, 0,0,:,:].astype(np.uint8)  # Adjust according to your model's output shape
    print("Mask shape:", pred_mask.shape)

    return pred_mask
