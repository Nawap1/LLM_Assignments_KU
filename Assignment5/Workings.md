# Image Chat Assistant - How It Works

## Overview
This is a Streamlit web application that allows users to upload images and have conversations about them using Google's Gemini AI model.

## Key Components

### 1. **Image Upload & Display**
- Users can upload images (JPG, JPEG, PNG) through a sidebar file uploader
- The uploaded image is displayed in the sidebar and stored in session state
- Only one image can be active at a time

### 2. **Chat Interface**
- Main chat area displays conversation history between user and AI assistant
- Users can type questions about the uploaded image in the chat input
- Messages are timestamped and styled with custom CSS for better UX

### 3. **AI Integration**
- Uses Google's Gemini 2.0 Flash model for image analysis and text generation
- The `get_gemini_response()` function sends both the image and user's text prompt to the model
- Includes error handling for API failures

### 4. **Session Management**
- Streamlit session state maintains:
  - Chat message history (`st.session_state.messages`)
  - Currently uploaded image (`st.session_state.current_image`)
- Users can clear the entire chat history and remove the current image

### 5. **User Experience Features**
- Loading spinner during AI response generation
- Error messages for missing images or API failures
- Clean, responsive layout with custom CSS styling
- Sidebar controls separate from main chat area

## Workflow
1. User uploads an image via sidebar
2. User types a question about the image
3. App sends image + question to Gemini AI
4. AI response is displayed in chat and saved to history
5. Process repeats for continued conversation about the image
