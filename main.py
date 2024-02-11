import os
from octo.model.octo_model import OctoModel
from PIL import Image
import requests
import jax
import matplotlib.pyplot as plt
import numpy as np

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
print("Start recording audio")
sample_name = "cha.m4a"
cmd = f'arecord -vv --format=cd --device={os.environ["AUDIO_INPUT_DEVICE"]} -r 48000 --duration=10 -c 1 {sample_name}'
print(cmd)
os.system(cmd)
print("Playing sound")
os.system(f"ffplay -nodisp -autoexit -loglevel quiet {sample_name}")

# Capture image
import cv2
camera_capture = cv2.VideoCapture(0)
rv, image = camera_capture.read()
print(f"Image Dimensions: {image.shape}")
img = np.array(Image.open(requests.get(image, stream=True).raw).resize((256, 256)))
plt.imshow(img)
img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "pad_mask": np.array([[True]])}
task = model.create_tasks(texts=["pick up the ball"])
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
print(action)   # [batch, action_chunk, action_dim]
camera_capture.release()
