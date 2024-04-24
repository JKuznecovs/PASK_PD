# Importē torch bibliotēku un treniņa tekstus no datini.py faila
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
from datini import dataset

# Modeļa valoda
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
# Kādi kategorijas pastāv
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: ["Recept", "Reminder", "Link", "Personal Info"].index(x)

# Pats klasificēšanas modulis
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Mašīnapmācības moduļi
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassificationModel(len(vocab), 64, 4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)

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
# Trenēšanas iestastījumi
epochs = 1000
for epoch in range(1, epochs + 1):
    model.train()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
torch.save(model, 'model.dat')
torch.save(vocab, 'vocab.pt')