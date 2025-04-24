import io
import os
# from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import aiohttp
from typing import Dict, Any

from graph import invoke_our_graph
import asyncio
from openai import OpenAI
from langsmith import Client
from langsmith.wrappers import wrap_openai
import openai
from st_callable_util_improved import get_streamlit_cb 
import numpy as np
import cv2

# load_dotenv()





# # Set up OpenAI API key from Streamlit secrets
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY not found in Streamlit secrets")


# Tumor Segmentation API URL
async def segment_tumor(base64_image: str, api_url: str = "https://pratiks-bhangale--tumor-segmentation-api.modal.run") -> Dict[str, Any]:
    """
    Send a base64-encoded image to the tumor segmentation API and get the results.
    
    Args:
        base64_image (str): Base64-encoded image string
        api_url (str): API endpoint URL (default: https://pratiks-bhangale--tumor-segmentation-api.modal.run)
        
    Returns:
        dict: Dictionary containing:
            - segmentation_image: Base64-encoded segmentation result
            - tumor_detection: Tumor detection result
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/predict",
            json={"image": base64_image},
            timeout=30
        ) as response:
            response.raise_for_status()
            return await response.json()


# Set the title of the Streamlit app
st.set_page_config(layout="wide")#, page_title="Llama 3 Model Document Q&A"
st.title("TumorVision AI")

# Create the Sidebar
sidebar = st.sidebar


# Get OpenAI API key from user input in sidebar
api_key = sidebar.text_input("Enter your OpenAI API key:", type="password")

# Validate API key
if not api_key:
    st.warning("Please enter your OpenAI API key to use this application.")
    openai_api_key = None
else:
    # Set the API key as environment variable
    os.environ["OPENAI_API_KEY"] = api_key
    openai_api_key = api_key


# Creating a Session State array to store and show a copy of the conversation to the user.
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize the messages list in session state

# Initialize images list in session state
if "images" not in st.session_state:
    st.session_state["images"] = []

# Initialize image in session state
if "image_base64" not in st.session_state:
    st.session_state["image_base64"] = None

# Initialize image in session state
if "segmentation_base64" not in st.session_state:
    st.session_state["segmentation_base64"] = None
    
# Initialize processed images tracking set
if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = set()

# Function to get image description from OpenAI
def get_image_description(image_base64, prompt=None, second_image_base64=None):
    """
    Get a description of the image(s) using OpenAI's GPT-4o model.
    
    Args:
        image_base64 (str): Base64-encoded image string
        prompt (str): The prompt to send to the model
        second_image_base64 (str, optional): Base64-encoded second image string
        
    Returns:
        str: The generated description
    """
    if prompt == None:
       prompt = "Describe this medical image in detail:"
    
    # if not st.session_state["openai_api_key"]:
    #     return "Please provide an OpenAI API key to analyze images."
    
    client = OpenAI()
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ]
    
    # Add second image if available
    if second_image_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{second_image_base64}"}})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )
    return response.choices[0].message.content




# Create the reset button for the chats
clear_chat = sidebar.button("Clear Chat")
if clear_chat:
    st.session_state["messages"] = []
    st.session_state["images"] = []
    st.session_state["image_base64"] = None
    st.session_state["segmentation_base64"] = None
    st.session_state["processed_images"] = set()  # Clear the processed images set
    st.toast('Conversation Deleted', icon='⚙️')


# Function to encode image to base64
def encode_image(image_file):
    if image_file is None:
        return None
    
    img = Image.open(image_file)
    
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        # Create a white background image
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Paste the image on the background using the alpha channel as mask
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str




# Image upload section
uploaded_image = sidebar.file_uploader(
    "Upload an image for analysis", 
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)







# Display the chat messages in the Streamlit app
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)








# Process uploaded image
if uploaded_image:

    
    # Convert image to base64
    img = Image.open(uploaded_image)
    

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    # Resize image to 256x256
    img = img.resize((256, 256), Image.LANCZOS)  # LANCZOS provides high-quality downsampling

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Store in session state
    st.session_state.image_base64 = img_str
    st.sidebar.success("Image uploaded successfully!")
    
    # Create a unique identifier for this image
    image_id = hash(st.session_state.image_base64)
    
    # Only process if we haven't seen this image before
    if image_id not in st.session_state.processed_images:
        try:
            
            with st.spinner('Retrieving segmentation Mask...'):
                # Create event loop and run the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(segment_tumor(st.session_state.image_base64))


            #f"Make sure to mention the result in the first line. Based on the segmentation mask, the tumor detection result is: {result['tumor_detection']}."
            prompt1 =f"""
            Make sure to mention the result in the first line. Based on the segmentation mask, the tumor detection result is: {result['tumor_detection']}.

            The tumor is highlighted in red pixels if it is detected.

            Please analyze the segmentation mask. Focus on:

            1. Location of abnormal tissue - identify the brain region (frontal, temporal, parietal, occipital lobe, cerebellum, brainstem) and whether it's unilateral or bilateral
            2. Potential differential diagnoses based on imaging characteristics (glioma, meningioma, metastasis, etc.)
            3. Mass effect - signs of compression on surrounding structures, midline shift, or ventricular compression
            4. Surrounding edema - presence and extent of edema (brain swelling)
            5. Enhancement pattern (if visible) - homogeneous or heterogeneous
            6. Any signs of hemorrhage, necrosis, or cystic components
            7. Impact on nearby critical structures
            8. Make sure to mention this is an approximation, refer to an actual medical professional for full diagnosis.

            Provide a concise summary that would assist a medical professional in treatment planning.
            """
            st.session_state.segmentation_base64 = result["segmentation_image"]

            # Convert the segmentation image from base64 to displayable format
            segmentation_image_bytes = base64.b64decode(result["segmentation_image"])
            segmentation_image = Image.open(io.BytesIO(segmentation_image_bytes))
            

            # Convert PIL images to numpy arrays for processing
            original_img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2RGB)  # Convert back to 3 channels for overlay
            segmentation_np = np.array(segmentation_image)

            # Create a colored mask (red overlay)
            colored_mask = np.zeros_like(original_img_np)
            colored_mask[segmentation_np > 0] = [255, 0, 0]  # Red color for the segmentation

            # Create the overlay
            overlay_img = cv2.addWeighted(original_img_np, 0.7, colored_mask, 0.3, 0)
            # Convert overlay image from numpy array to base64
            overlay_img_pil = Image.fromarray(overlay_img)
            overlay_buffer = BytesIO()
            overlay_img_pil.save(overlay_buffer, format="JPEG")
            overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode()

            with st.spinner('Generating image descriptions...'):
                # Get descriptions of both images using OpenAI
                original_image_description = get_image_description(
                    image_base64=overlay_base64,
                    # second_image_base64=overlay_base64,
                    prompt=prompt1
                )
                
            
            # Add the message to chat history
            st.session_state.messages.append(AIMessage(content=original_image_description))
            
            # Create the message content with images directly displayed here
            with st.chat_message("AI"):
                st.write(original_image_description)
                
                # Display original and segmented images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original Brain Scan")
                with col2:
                    st.image(overlay_img, caption="Segmentation Result")
            
            # Mark this image as processed
            st.session_state.processed_images.add(image_id)
            

        except aiohttp.ClientError as e:
            error_message = f"Error during image analysis: {str(e)}"
            st.session_state["messages"].append(AIMessage(content=error_message))
            with st.chat_message("AI"):
                st.write(error_message)
 











# # Display the chat messages in the Streamlit app
# for message in st.session_state.messages:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.write(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.write(message.content)







question = st.chat_input(placeholder="Ask Anything.", key=1, accept_file=True, file_type=[".png", ".jpg", ".jpeg"])


if question:

    # Add the user question to the session state messages
    st.session_state["messages"].append(HumanMessage(content=question.text))

    # Update the image processing section in your code:
    with st.chat_message("Human"):
        st.write(question.text)
        
        # Process and display uploaded images
        uploaded_images = []
        if question and question.files:
            # Add the image to the session state messages                
            for file in question.files:
                # st.session_state["messages"].append(HumanMessage(content={"type": "image", "image_data": encode_image(file)}))
                st.image(file)
                encoded_image = encode_image(file)
                if encoded_image:
                    uploaded_images.append(encoded_image)

    # Invoke the graph with the user question, images, and the callback handler
    with st.chat_message("AI"):
        st_callback = get_streamlit_cb(st.container())
        with st.spinner('Generating response...'):  # Add spinner here
            response = invoke_our_graph(
                # api_key=st.session_state["openai_api_key"],
                st_messages=st.session_state.messages,
                images=uploaded_images if uploaded_images else None,
                callables=[st_callback]
            )
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))
