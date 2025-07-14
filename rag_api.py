from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# Global variables for models and data
embedding_model = None
gpu_index = None
metadata = None
llm_model = None
tokenizer = None
conversation_history = {}

def initialize_models():
    """Initialize all models and load data"""
    global embedding_model, gpu_index, metadata, llm_model, tokenizer
    
    print("Initializing models...")
    
    # Ensure GPU is used if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available. Using CPU, which may be slower.")
    
    # Initialize the BGE-M3 model for embeddings
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', device=device)
    
    # Load the FAISS index
    if os.path.exists("subject_embeddings.faiss"):
        cpu_index = faiss.read_index("subject_embeddings.faiss")
        
        # Move index to GPU if available
        if device.type == "cuda":
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            gpu_index = cpu_index  # Fallback to CPU index
    else:
        print("Error: subject_embeddings.faiss not found!")
        return False
    
    # Load metadata
    if os.path.exists("subject_metadata.pkl"):
        with open("subject_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
    else:
        print("Error: subject_metadata.pkl not found!")
        return False
    
    # Initialize the Qwen3-0.6B model and tokenizer
    llm_model_name = "Qwen/Qwen3-0.6B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        print("Using fallback response generation...")
        llm_model = None
        tokenizer = None
    
    print("Models initialized successfully!")
    return True

def search_faiss(query_text: str, top_k: int = 3) -> List[Dict]:
    """Search FAISS index for relevant documents"""
    if embedding_model is None or gpu_index is None or metadata is None:
        return []
    
    query_embedding = embedding_model.encode(query_text, max_length=256)['dense_vecs']
    query_embedding = np.array([query_embedding]).astype('float32')
    
    search_k = top_k
    distances, indices = gpu_index.search(query_embedding, search_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(metadata):
            continue
        result = {
            'index': int(idx),
            'distance': float(distances[0][i]),
            'subject': metadata[idx]['subject'],
            'text': metadata[idx]['text']
        }
        results.append(result)
        if len(results) >= top_k:
            break
    
    return results

def call_llm(prompt: str, history: List[Dict[str, str]] = [], max_new_tokens: int = 200) -> str:
    """Call the Qwen3-0.6B model with chat template"""
    if llm_model is None or tokenizer is None:
        # Fallback response if model is not available
        return "عذراً، لا أستطيع الإجابة على سؤالك في الوقت الحالي. يرجى المحاولة مرة أخرى لاحقاً."
    
    try:
        # Prepare chat history
        messages = []
        for turn in history[-3:]:
            messages.append({"role": "user", "content": turn['user']})
            messages.append({"role": "assistant", "content": turn['assistant']})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
        
        # Generate response
        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Decode the response
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        return "عذراً، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى."

def conversational_rag(query: str, session_id: str = "default") -> Dict[str, str]:
    """Main RAG function with conversation history"""
    # Get or create conversation history for this session
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    history = conversation_history[session_id]
    
    # Search for relevant context
    results = search_faiss(query, top_k=3)
    context = "\n".join([f"- {result['text']} (موضوع: {result['subject']})" for result in results])
    
    # Create prompt for child-friendly response
    prompt = f"""
أنا معلم ودود أشرح المواد المدرسية للطلاب من الصف الأول إلى الصف الثاني عشر. سأقدم الإجابات باللغة العربية فقط بأسلوب بسيط وودود يناسب الأطفال من عمر 6-12 سنة. أستخدم الرموز التعبيرية والأمثلة البسيطة.

السياق (استخدمه إذا كان مفيدًا):
{context}

السؤال: {query}

أجب بالعربية الفصحى بطريقة ودودة وسهلة الفهم تناسب الأطفال، واستخدم الرموز التعبيرية لجعل الإجابة ممتعة! 🌟
"""
    
    response = call_llm(prompt, history)
    
    # Update conversation history
    history.append({"user": query, "assistant": response})
    conversation_history[session_id] = history[-10:]  # Keep last 10 exchanges
    
    return {"answer": response, "context_used": len(results) > 0}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "AnnanScience RAG API is running!"})

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request"}), 400
        
        user_message = data['message']
        session_id = data.get('session_id', 'default')
        
        # Get response from RAG system
        result = conversational_rag(user_message, session_id)
        
        return jsonify({
            "answer": result["answer"],
            "context_used": result["context_used"],
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى.",
            "details": str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if session_id in conversation_history:
            del conversation_history[session_id]
        
        return jsonify({"message": "تم إعادة تعيين المحادثة بنجاح!"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize models on startup
    if initialize_models():
        print("Starting AnnanScience RAG API server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize models. Please check your files.")
