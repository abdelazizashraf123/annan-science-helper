import unsloth
from unsloth import FastLanguageModel
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import torch
from transformers import AutoTokenizer
from typing import List, Dict
import pickle
import os
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("Warning: CUDA not available. Using CPU, which may be slower.")

# Initialize the BGE-M3 model for embeddings
embedding_model = BGEM3FlagModel('BAAI/bge-m3', device=device)

# Load the FAISS index
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cpu_index = faiss.read_index(os.path.join(base_dir, 'data', 'subject_embeddings.faiss'))

# Move index to GPU if available
if device.type == "cuda":
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
else:
    gpu_index = cpu_index

# Load metadata
metadata_path = os.path.join(base_dir, 'data', 'subject_metadata.pkl')
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# Initialize the Qwen2.5-3B-Instruct model and tokenizer with Unsloth
llm_model_name = "Qwen/Qwen2.5-3B-Instruct"
llm_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=llm_model_name,
    max_seq_length=2048,
    device_map="auto"
)

# Initialize LangChain memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="query",
    output_key="content",
    return_messages=True
)

# Define the prompt template with memory
prompt_template = PromptTemplate(
    input_variables=["chat_history", "query", "context"],
    template="""انت معلم ذكي وودود، وظيفتك مساعدة الطلاب من جميع المراحل الدراسية العمرية عن طريق الرد على أسئلتهم في مواضيع الرياضيات، الأحياء، الكيمياء، الفيزياء، البيئة، اللغة العربية، والعلوم بطريقة بسيطة وسهلة وواضحة تتناسب مع مستواهم العمري وفقًا للسياق المقدم. يجب استخدام اللغة العربية فقط في الإجابة على أسئلة الطلاب، مع مراعاة تقديم الإجابات بشكل دقيق ومناسب للسياق.

# تعليمات إضافية
* تأكد من أن الإجابة خالية من التعقيدات ومباشرة.
* إذا كان السؤال يتطلب شرحًا علميًا، استخدم لغة مبسطة تناسب المرحلة العمرية للطالب.
* لا تستخدم مصطلحات معقدة إلا إذا كانت ضرورية، وفي هذه الحالة اشرحها ببساطة.
* إذا كان هناك سجل محادثة سابق، استخدمه لضمان استمرارية السياق.

سجل المحادثة السابقة:
{chat_history}

السؤال: 
{query}

السياق:
{context}"""
)

def search_faiss(query_text: str, index, embedding_model, metadata: List[Dict], top_k: int = 3) -> List[Dict]:
    query_embedding = embedding_model.encode(query_text, max_length=256)['dense_vecs']
    query_embedding = np.array([query_embedding]).astype('float32')
    search_k = top_k
    distances, indices = index.search(query_embedding, search_k)
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

def call_llm(prompt: str, max_new_tokens: int = 512) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    generated_ids = llm_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content

def conversational_rag(query: str, session_id: str) -> Dict[str, str]:
    # Use session-specific memory
    if not hasattr(conversational_rag, 'memory_dict'):
        conversational_rag.memory_dict = {}
    if session_id not in conversational_rag.memory_dict:
        conversational_rag.memory_dict[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="query",
            output_key="content",
            return_messages=True
        )
    session_memory = conversational_rag.memory_dict[session_id]
    
    results = search_faiss(query, gpu_index, embedding_model, metadata, top_k=3)
    context = "\n".join([f"- {result['text']} (موضوع: {result['subject']})" for result in results])
    chat_history = session_memory.load_memory_variables({})["chat_history"]
    prompt = prompt_template.format(
        chat_history=chat_history,
        query=query,
        context=context
    )
    response = call_llm(prompt)
    session_memory.save_context({"query": query}, {"content": response})
    return {"content": response}