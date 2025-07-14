# AnnanScience - Setup Guide

## Overview
AnnanScience is a child-friendly Arabic science chat interface that connects to your RAG (Retrieval-Augmented Generation) model. It's designed for children aged 6-12 and provides educational content in Arabic.

## Files Structure
```
├── index.html          # Main HTML interface
├── style.css           # Child-friendly styling
├── app.js              # Frontend JavaScript
├── rag_api.py          # Flask API server for RAG model
├── requirements.txt    # Python dependencies
├── encode_of_data.ipynb # Your original RAG notebook
├── subject_embeddings.faiss # FAISS vector database
├── subject_metadata.pkl     # Metadata for embeddings
└── SETUP_GUIDE.md      # This file
```

## Setup Instructions

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Required Files Exist
Make sure these files are in your project directory:
- `subject_embeddings.faiss` (your FAISS index)
- `subject_metadata.pkl` (your metadata file)

### 3. Start the RAG API Server
```bash
python rag_api.py
```

The server will start on `http://localhost:5000`

### 4. Start the Frontend
Open a new terminal and start a simple HTTP server:
```bash
python -m http.server 8000
```

### 5. Access the Application
Open your browser and go to: `http://localhost:8000`

## Features

### Frontend Features
- 🎨 Child-friendly colorful interface
- 🧪 Science-themed design with emojis
- 💬 Real-time chat with your RAG model
- 🔄 Conversation reset functionality
- ⚙️ Settings panel for API configuration
- 📱 Responsive design for mobile and desktop

### Backend Features
- 🤖 Integration with your existing RAG model
- 🔍 FAISS vector search for relevant context
- 💾 Conversation history management
- 🌐 RESTful API endpoints
- 🛡️ Error handling and fallback responses

## API Endpoints

### POST /chat
Send a message to the RAG model
```json
{
  "message": "ما هي الخلية؟",
  "session_id": "optional_session_id"
}
```

Response:
```json
{
  "answer": "الخلية هي أصغر وحدة في الكائن الحي...",
  "context_used": true,
  "session_id": "session_123"
}
```

### POST /reset
Reset conversation history
```json
{
  "session_id": "session_123"
}
```

### GET /health
Health check endpoint

## Customization

### Changing the System Prompt
The system prompt is configured in the RAG API to be child-friendly and Arabic-focused. You can modify it in `rag_api.py` in the `conversational_rag` function.

### Styling
Modify `style.css` to change colors, fonts, or layout. The current design uses:
- Comic Sans MS font for child-friendliness
- Bright, cheerful colors
- Rounded corners and soft shadows
- Emoji integration

### Adding Features
You can extend the functionality by:
1. Adding new endpoints to `rag_api.py`
2. Updating the frontend JavaScript in `app.js`
3. Modifying the HTML structure in `index.html`

## Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure you have sufficient RAM/VRAM
   - Check if CUDA is available for GPU acceleration
   - Verify all model files are accessible

2. **FAISS index errors**
   - Ensure `subject_embeddings.faiss` exists
   - Check file permissions
   - Verify the index was created correctly

3. **API connection errors**
   - Ensure the Flask server is running on port 5000
   - Check firewall settings
   - Verify the API endpoint in settings

4. **Frontend not loading**
   - Ensure the HTTP server is running on port 8000
   - Check browser console for JavaScript errors
   - Verify all files are in the correct directory

### Performance Tips

1. **For better performance:**
   - Use GPU acceleration if available
   - Adjust `max_new_tokens` in the API for shorter responses
   - Limit conversation history length

2. **For production deployment:**
   - Use a proper WSGI server like Gunicorn
   - Add authentication if needed
   - Implement rate limiting
   - Add logging and monitoring

## Development

### Adding New Subjects
To add new subjects to your RAG model:
1. Update your data processing in the Jupyter notebook
2. Regenerate the FAISS index and metadata
3. Restart the API server

### Modifying the Chat Interface
The chat interface can be customized by editing:
- `index.html` for structure
- `style.css` for appearance
- `app.js` for functionality

## Support

If you encounter issues:
1. Check the console logs in both browser and terminal
2. Verify all dependencies are installed correctly
3. Ensure your RAG model files are properly formatted
4. Test the API endpoints directly using curl or Postman

## License

This project is designed for educational purposes and child-friendly science learning.
