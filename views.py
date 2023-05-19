from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from django.shortcuts import redirect, render
import re
from django.contrib.auth import authenticate, login, logout
from .models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import torch
from django.views.generic import View

@csrf_exempt
def login_view(request):
    if request.method == "POST":
        data = request.POST
        print(data)
        username = data.get("email")
        password = data.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("index")
        else:
            return JsonResponse(
                {"status": "error", "message": "Invalid login credentials"}, status=400
            )


@csrf_exempt
def signup_view(request):
    if request.method == "POST":
        data = request.POST
        email = data.get("email")
        password = data.get("password")
        name = data.get("name")
        if not User.objects.filter(email=email).exists():
            if not email or not password or not name:
                return JsonResponse(
                    {"status": "error", "message": "All fields are required"},
                    status=400,
                )
            User.objects.create_user(
                username=email, email=email, password=password, name=name
            )
            return JsonResponse({"status": "success"}, status=200)
        else:
            return JsonResponse(
                {"status": "error", "message": "User already exists"}, status=400
            )


class IndexView(APIView):
    def get(self, request, *args, **kwargs):
        min_length = 100  # Set your desired default value
        max_length = 128  # Set your desired default value
        return render(
            request,
            "api/index.html",
            {
                "min_length": min_length,
                "max_length": max_length,
                "is_authenticated": request.user.is_authenticated,
            },
        )


class LogoutView(View):
    def get(self, request, *args, **kwargs):
        logout(request)
        return redirect("index")



class SummarizeView(APIView):
    WHITESPACE_HANDLER = lambda self, value: re.sub(
        "\s+", " ", re.sub("\n+", " ", value.strip())
    )
    model = None
    tokenizer = None
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Set the device

    @classmethod
    def load_model(cls):
        if cls.model is None or cls.tokenizer is None:
            model_name = "nepali_sum_model/finetuned-model"
            cls.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device=cls.device
            )  # Specify the device
            cls.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
                cls.device
            )  # Specify the device

    def post(self, request, *args, **kwargs):
        # if not request.user.is_authenticated:
        #     return Response({'error': 'User is not authenticated.'}, status=401)

        input_text = request.data.get("text")
        # min_length = int(
        #     request.data.get("min_length")
        # )  # Get the min_length value from the request
        # max_length = int(
        #     request.data.get("min_length")
        # )  # Get the max_length value from the request

        self.load_model()  # Load the model if not already loaded

        input_ids = self.tokenizer(
            [self.WHITESPACE_HANDLER(input_text)],
            return_tensors="pt",
            # padding="max_length",
            truncation=True,
            max_length=512,
        )["input_ids"].to(
            self.device
        )  # Move the input_ids tensor to the device

        output_ids = self.model.generate(
            input_ids=input_ids,
            # min_length=min_length,  # Use the retrieved min_length value
            # max_length=max_length,  # Use the retrieved max_length value
            no_repeat_ngram_size=2,
            num_beams=4,
        )[0]

        summary = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return Response({"summary": summary})


# def clean_str(text):
#     try:
#         text = re.sub(
#             r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', '', text)
#         text = re.sub(r'[|:}{=]', ' ', text)
#         text = re.sub(r'[;]', ' ', text)
#         text = re.sub(r'[\n]', ' ', text)
#         text = re.sub(r'[\t]', ' ', text)
#         text = re.sub(r'[-]', ' ', text)
#         text = re.sub(r'[+]', ' ', text)
#         text = re.sub(r'[*]', ' ', text)
#         text = re.sub(r'[/]', ' ', text)
#         text = re.sub(r'[//]', ' ', text)
#         text = re.sub(r'[@]', ' ', text)
#         text = re.sub(r'[,]', ' ', text)
#         text = re.sub(r'[)]', ' ', text)
#         text = re.sub(' +', ' ', text)
#         # text = re.sub('\n+', '\n', text)
#         # text = re.sub('\t+', '\t', text)
#         # text = re.sub('\n+', '\n', text)
#         text = re.sub(r'[-]', ' ', text)
#         text = re.sub(r'[(]', ' ', text)
#         text = re.sub(' + ', ' ', text)
#         # text = text.encode('ascii', errors='ignore').decode("utf-8")
#         return text
#     except Exception as e:
#         print(e)
#         print(f"Error while cleaning text --->{text}")
#         pass

# import string


# class InputValidator:
#     def __init__(self, file_content):
#         self.file_content = file_content
#         self.text = ''
#         self.eng_text = ''

#     def detect_language(self,character):
#         maxchar = max(character)
#         if (u'\u0900' <= maxchar <= u'\u097f') or (maxchar == u"\u0020"):
#             return True

#     def validate_to_var(self):
#         # print(self.file_content)
#         for index, i in enumerate(self.file_content):
#             # for i in j:
#             # print(i)
#             isNep = self.detect_language(i)
#             if isNep == True:
#                 # print(i,end="\t")
#                 # print(isEng)
#                 self.text = self.text + i
#             else:
#                 self.eng_text = self.eng_text + i
#             # print(self.text)
#                 # print(f"not nepali {index} : {self.eng_text.encode()}")
#         # print(f"not nepali : {self.eng_text}")
#         return self.text
#     """
#     appends all the english texts to a file
#     """
#     def validate_to_file(self):
#         for i in self.file_content:
#             isEng = self.detect_language(i)
#             if isEng == True:
#                 with open('out.txt','a', encoding='utf-8') as f:
#                     f.write(i)
