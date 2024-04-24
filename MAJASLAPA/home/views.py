from django.shortcuts import render
from .models import Post
from django.core.exceptions import ValidationError
import torch
from torchtext.data.utils import get_tokenizer
from .torch_models import TextClassificationModel
vocab = torch.load('vocab.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('basic_english')
model = TextClassificationModel(len(vocab), 64, 4).to(device)

def classify_new_text(new_text, model, vocab, tokenization_func):
    tokens = tokenization_func(new_text)
    processed_text = torch.tensor(vocab(tokens), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        predicted_label = model(processed_text.unsqueeze(0), None)
    category_index = predicted_label.argmax(1).item()
    categories = ["Recept", "Reminder", "Link", "Personal Info"]
    predicted_category = categories[category_index]
    
    return predicted_category

def home(request):
    error = None
    if request.method == 'POST':
        title = request.POST.get('title')
        text = request.POST.get('text')

        if not title or not text:
            error = 'Virsraksts vai teksts nevar būt tukšs!!!'
        else:
            post = Post()
            post.title = title
            post.text = text
            new_text = post.text
            predicted_category = classify_new_text(new_text, model, vocab, tokenizer)
            post.type = predicted_category
            post.save()

    all_posts = Post.objects.all()

    context = {
        "posts": all_posts,
        "error": error,
    }

    return render(request, 'home/homepage.html', context)