from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .utils import conversational_rag

def chatbot_view(request):
    messages = []
    if 'conversation' not in request.session:
        request.session['conversation'] = []
        messages.append({
            'role': 'assistant',
            'content': 'مرحبًا يا عالم صغير! 👋 أنا AnnanScience، معلمك العلمي الودود! اسألني أي شيء عن العلم - من لماذا السماء زرقاء 🌌 إلى كيف تنمو النباتات 🌱! ما الذي تريد استكشافه اليوم؟ 🔍'
        })
    else:
        messages = request.session['conversation']

    if request.method == 'POST':
        query = request.POST.get('query', '')
        session_id = request.session.session_key or request.session.get('session_id')
        if not session_id:
            request.session.create()
            session_id = request.session.session_key
        if query.lower() == 'خروج':
            request.session['conversation'] = []
            return render(request, 'chatbot/index.html', {
                'messages': [{'role': 'assistant', 'content': 'تم إنهاء المحادثة.'}],
                'loading': False
            })
        
        # Add user message
        messages.append({'role': 'user', 'content': query})
        # Get response
        response = conversational_rag(query, session_id)
        messages.append({'role': 'assistant', 'content': response['content']})
        request.session['conversation'] = messages
        return render(request, 'chatbot/index.html', {
            'messages': messages,
            'loading': False
        })

    return render(request, 'chatbot/index.html', {
        'messages': messages,
        'loading': False
    })

@csrf_exempt
def query_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data.get('query', '')
        session_id = request.session.session_key or request.session.get('session_id')
        if not session_id:
            request.session.create()
            session_id = request.session.session_key
        if query.lower() == 'خروج':
            return JsonResponse({'content': 'تم إنهاء المحادثة.'})
        response = conversational_rag(query, session_id)
        return JsonResponse(response)
    return JsonResponse({'error': 'Invalid request'}, status=400)