# streamlit_picture_generator

## What does it do?
This project aims at generating pictures according to the text prompt or to the picture that you upload. It uses a simple pretrained diffusion model ZeroCool94/stable-diffusion-v1-5 that can be substituted with a different one if needed. It works completely locally, so it can be run safely. It has 2 modules: one gor Text-to-Image generation and the other one for Image-to-Image generation (where the uploaded picture is used for enriching the text prompt. It can create simple and funny images. You can choose the number of inference steps, the prompt guidance scale, the negative prompt before each generation and how much the initial picture can change. After generating you can download the result in .png format and enjoy it :)

## How do I start the project?
- Download the file called app.py from this repository or copy it
- Open your terminal, go to the folder with this file and create a virtual environment using python -m venv name
- Activate the environment using name\Scripts\activate (Windows) or name/Scripts/activate (Linux)
- Install all the required packages using pip install streamlit torch diffusers transformers accelerate torchvision
- Print streamlit wun app.py
- A localhost page will automatically open in your browser. The port is 8001 or it can automatically reach another port if 8001 is taken.
- Have fun using simple UI!

## You can see the examples of generation in this repository too.
<img width="512" height="512" alt="cheese with guinea pigs in it _20" src="https://github.com/user-attachments/assets/b400b0a0-63c6-468d-b1c3-d0b476b6122e" />


# Note!
Mind that the speed of generation depends on your computer, its graphics processor. The quality and latency can vary on different machines.
