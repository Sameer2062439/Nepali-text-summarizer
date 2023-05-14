from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from django.shortcuts import render
import re
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import torch

@csrf_exempt
def login_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        username = data.get('email')
        password = data.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({"status": "success"}, status=200)
        else:
            return JsonResponse({"status": "error", "message": "Invalid login credentials"}, status=400)

@csrf_exempt
def signup_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        username = data.get('email')
        password = data.get('password')
        name = data.get('name')
        if not User.objects.filter(username=username).exists():
            User.objects.create_user(username, username, password, first_name=name)
            return JsonResponse({"status": "success"}, status=200)
        else:
            return JsonResponse({"status": "error", "message": "User already exists"}, status=400)


class IndexView(APIView):
    def get(self, request, *args, **kwargs):
        min_length = 100  # Set your desired default value
        max_length = 128  # Set your desired default value
        return render(request, 'api/index.html', {'min_length': min_length, 'max_length': max_length})

class SummarizeView(APIView):
    WHITESPACE_HANDLER = lambda self, value: re.sub('\s+', ' ', re.sub('\n+', ' ', value.strip()))
    model = None
    tokenizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device

    @classmethod
    def load_model(cls):
        if cls.model is None or cls.tokenizer is None:
            model_name = "nepali_sum_model/finetuned-model"
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name, device=cls.device)  # Specify the device
            cls.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(cls.device)  # Specify the device

    def post(self, request, *args, **kwargs):
        # if not request.user.is_authenticated:
        #     return Response({'error': 'User is not authenticated.'}, status=401)

        input_text = request.data.get('text')
        min_length = int(request.data.get('min_length'))  # Get the min_length value from the request
        max_length = int(request.data.get('min_length'))  # Get the max_length value from the request

        self.load_model()  # Load the model if not already loaded

        input_ids = self.tokenizer(
            [self.WHITESPACE_HANDLER(input_text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"].to(self.device)  # Move the input_ids tensor to the device

        output_ids = self.model.generate(
            input_ids=input_ids,
            min_length=min_length,  # Use the retrieved min_length value
            max_length=max_length,  # Use the retrieved max_length value
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]

        summary = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return Response({'summary': summary})
