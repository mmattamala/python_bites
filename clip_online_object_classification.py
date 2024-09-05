#!/usr/bin/env python
# Author: Matias Mattamala
# Description: Online CLIP object classification demo
# Disclaimer: OpenCV and plotting code from https://medium.com/@Mert.A/real-time-plotting-with-opencv-and-matplotlib-2a452fbbbaf9
#
# Dependencies (assuming ROS installation on base system)
#  python3 -m venv env
#  source env/bin/activate
#  pip install opencv-python ftfy regex tqdm matplotlib
#  pip install torch torchvision
#  pip install git+https://github.com/openai/CLIP.git

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import clip
from PIL import Image


CLASSES_FILE = "assets/semantic_classes.txt"

if __name__ == "__main__":
    # Choose backend
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device [{device}]")

    # Load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load semantic classes from file
    with open(CLASSES_FILE, "r") as file:
        classes = file.read().split("\n")
        classes = classes[:-1]
    text = clip.tokenize(classes).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text)

    # Configure plot
    fig = plt.figure(figsize=(6.4, 3.6), dpi=100)

    fig, axs = plt.subplots(2, 1, figsize=(6.4, 3.6), dpi=100, sharex=True)
    cap = cv2.VideoCapture(1)

    # Preallocate array for Bayesian prior
    N = len(classes)
    prior = np.ones(N) / N

    # Main loop
    while True:
        # Clear axes
        for ax in axs:
            ax.clear()

        # Read image
        success, img = cap.read()
        img = cv2.resize(img, (640, 360))
        image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)

        with torch.inference_mode():
            image_features = model.encode_image(image)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # Bayes filter
        belief = np.multiply(probs, prior)
        posterior = belief / np.sum(belief)

        # Get most likely class
        best_idx = int(np.argmax(posterior))
        best_class_prob = posterior[best_idx]
        best_class = classes[best_idx]

        # Update prior for next iteration
        prior = posterior.copy()

        # Plotting fps data
        axs[0].bar(classes, probs)
        axs[0].set_ylabel("Measurement")

        # Plot
        axs[1].bar(classes, prior, color="r")
        axs[1].set_ylabel("Posterior")
        axs[1].set_xlabel("Classes")
        axs[1].set_title(f"Best class: {best_class} ({best_class_prob:.2f})")
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Convert matplotlib into image
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        buf = np.asarray(buf)
        plot = np.array(Image.fromarray(buf).convert("RGB"))
        plot = cv2.cvtColor(plot, cv2.COLOR_RGBA2BGR)
        plot = cv2.resize(plot, (640, 360))

        # Combining Original Frame and Plot Image
        result_img = np.hstack([img, plot])

        # Displaying the Combined Image:
        cv2.imshow("Image", result_img)
        cv2.waitKey(1)

    cap.close()
