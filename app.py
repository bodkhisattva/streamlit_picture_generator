import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline 
from PIL import Image
from io import BytesIO
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor


@st.cache_resource
def load_t2i_model():
    model_id = "ZeroCool94/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
    
    pipe = StableDiffusionPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        safety_checker=None,  
        feature_extractor=feature_extractor,  
        requires_safety_checker=False
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    return pipe

@st.cache_resource
def load_i2i_model():
    model_id = "ZeroCool94/stable-diffusion-v1-5"
    
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
    
    pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=feature_extractor,
        requires_safety_checker=False
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    return pipe

st.title("Image generator (Text-to-Image and Image-to-Image)")

mode = st.radio("Choose:", ["Text-to-Image)", "Image-to-Image"])

prompt = st.text_input("Your request:", placeholder="An example: a cat-astronaut playing with the planet Earth'")

with st.sidebar:
    st.header("Options")
    num_inference_steps = st.slider("Number of steps", 10, 50, 30)
    guidance_scale = st.slider("How much do you want me to follow your prompt?", 1.0, 15.0, 7.5)
    negative_prompt = st.text_input("What should I exclude (optional)?", placeholder="uneven lines")

    init_image = None
    strength = 0.75
    if mode == "Image-to-Image":
        st.divider()
        st.subheader("Image Options")
        uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            init_image = Image.open(uploaded_file).convert("RGB")
            st.image(init_image, caption="Initial image", use_container_width=True)
        
        strength = st.slider(
            "Degree of influence", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.75,
            help="It decides how much the initial picture can change. 0.0 - no changes, 1.0 - - a while transformation."
        )

if st.button("Generate now!!") and prompt:
    with st.spinner("Generating an image...this might take a while. While waiting, consider puschasing a better graphic processor."):
        result_image = None
        
        if mode == "Text-to-Image":
            pipe = load_t2i_model()
            result_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        elif mode == "Image-to-Image":
            if init_image is None:
                st.error("Upload your image please.")
                st.stop()
            init_image = init_image.resize((512, 512)) 
                
            pipe = load_i2i_model()
                    
            result_image = pipe(
                prompt=prompt,
                image=init_image, 
                strength=strength, 
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        if result_image:
            st.image(result_image, caption=f"Результат: {prompt}", use_container_width=True)
            
            buf = BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button(
                label="Download",
                data=buf.getvalue(),
                file_name=f"{prompt}[:20].png",
                mime="image/png"
            )
