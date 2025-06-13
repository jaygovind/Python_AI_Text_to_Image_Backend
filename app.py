import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import imageio

# Load model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Main processing function
def generate_3d_video(input_img):
    img = np.array(input_img.convert("RGB"))
    H, W = img.shape[:2]

    transformed = midas_transforms(input_img).to(device)
    with torch.no_grad():
        depth = midas(transformed).squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    frames = []
    for shift in np.linspace(-30, 30, num=60):
        map_x, map_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (map_x + shift * (1 - depth)).astype(np.float32)
        map_y = map_y.astype(np.float32)

        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames.append(warped)

    gif_path = "3d_parallax.gif"
    imageio.mimsave(gif_path, frames, fps=24)
    return gif_path

# Gradio UI
iface = gr.Interface(
    fn=generate_3d_video,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="filepath", label="3D Parallax GIF"),
    title="3D Video from Image",
    description="Upload a 2D image and get a 3D parallax animation using depth estimation (MiDaS)."
)

iface.launch()
