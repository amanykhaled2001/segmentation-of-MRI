import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess
from predict import predict

def inference(filepath):
    input_batch = preprocess(filepath)
    result = predict(input_batch)
    pred_mask = np.array(result).astype(np.uint8)  # Convert to uint8 for proper image display
    plt.imshow(pred_mask, cmap='gray')  # Display the mask as grayscale
    plt.title("Predicted Tumor Mask")
    plt.axis('off')  # Turn off axis
    plt.close()  # Close the plot to avoid displaying it in Gradio interface

    return pred_mask

title = "Brain MRI Tumor Detection - Semantic Segmentation using PyTorch"
description = "Segmentation of tumor areas from Brain MRI images"
article = "<p style='text-align: center'><a href='https://www.kaggle.com/s0mnaths/brain-mri-unet-pytorch/' target='_blank'>Kaggle Notebook: Brain MRI-UNET-PyTorch</a> | <a href='https://github.com/s0mnaths/Brain-Tumor-Segmentation' target='_blank'>Github Repo</a></p>"
examples = [
    ['/content/drive/MyDrive/Brain-Tumor-Segmentation/test-samples/TCGA_CS_4941_19960909_15.png'],
    ['/content/drive/MyDrive/Brain-Tumor-Segmentation/test-samples/TCGA_CS_4942_19970222_11.png'],
    ['/content/drive/MyDrive/Brain-Tumor-Segmentation/test-samples/TCGA_CS_4942_19970222_12.png'],
    ['/content/drive/MyDrive/Brain-Tumor-Segmentation/test-samples/TCGA_CS_4941_19960909_15.png']
]

app= gr.Interface(inference, inputs=gr.Image(type="filepath"), outputs=gr.Image('plot'), title=title,
            description=description,
            article=article,
            examples=examples)
app.queue()
app.launch(debug=True,share=True)