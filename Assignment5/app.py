import streamlit as st
import PIL.Image
import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

def get_gemini_response(images, prompt):
    """
    Generate a response from Gemini model based on images and prompt
    Args:
        images: List of PIL Image objects
        prompt: String containing the user's question
    Returns:
        String containing the model's response or None if error occurs
    """
    try:
        content = [prompt] + images
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

def display_chat_message(role, content, timestamp=None):
    """
    Display a chat message with the specified role and content
    Args:
        role: String indicating message sender ('user' or 'assistant')
        content: String containing message content
        timestamp: Optional timestamp for the message
    """
    with st.chat_message(role):
        st.write(content)
        if timestamp:
            st.caption(timestamp)

def main():
    """Main application function"""
    # Configure page settings
    st.set_page_config(page_title="Image Chat Assistant", layout="wide")
    initialize_session_state()

    # Apply custom CSS for chat styling
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stChatMessage[data-testid="stChatMessage-USER"] {
            background-color: #e6f3ff;
        }
        .stChatMessage[data-testid="stChatMessage-ASSISTANT"] {
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main application layout
    st.title("💬 Image Chat Assistant")
    
    # Sidebar for image upload and controls
    with st.sidebar:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.session_state.current_image = image
            
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_image = None

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("timestamp")
        )

    # Chat input and response generation
    if prompt := st.chat_input("Ask about the image..."):
        if not st.session_state.current_image:
            st.error("Please upload an image first!")
            return

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.strftime("%H:%M")
        })
        display_chat_message("user", prompt)

        with st.spinner("Thinking..."):
            response = get_gemini_response([st.session_state.current_image], prompt)
            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": time.strftime("%H:%M")
                })
                display_chat_message("assistant", response)

if __name__ == "__main__":
    main()
