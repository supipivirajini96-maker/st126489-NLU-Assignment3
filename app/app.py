import math
import torch
import pickle
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import nltk
from nltk.tokenize import word_tokenize

from torchtext.vocab import build_vocab_from_iterator

from model import Encoder, Decoder, Seq2SeqTransformer

# --------------------------------------------------
# NLTK
# --------------------------------------------------
nltk.download("punkt")

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Language settings
# --------------------------------------------------
SRC_LANGUAGE = "en"
TRG_LANGUAGE = "si"

# Special tokens (MUST match training)
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# --------------------------------------------------
# Tokenizer (same as training)
# --------------------------------------------------
def tokenize(text):
    return word_tokenize(text.lower())

# --------------------------------------------------
# Load training data (for vocab ONLY)
# --------------------------------------------------
def load_parallel_data(en_path, si_path):
    with open(en_path, encoding="utf-8") as f:
        en_lines = f.readlines()
    with open(si_path, encoding="utf-8") as f:
        si_lines = f.readlines()
    return list(zip(en_lines, si_lines))

train_data = load_parallel_data(
    "splits/train_en.txt",
    "splits/train_si.txt"
)

# --------------------------------------------------
# Build vocab WITHOUT unsupported args (OPTION 2)
# --------------------------------------------------
def yield_tokens(data, lang):
    for src, trg in data:
        if lang == SRC_LANGUAGE:
            yield tokenize(src)
        else:
            yield tokenize(trg)

SRC_VOCAB = build_vocab_from_iterator(yield_tokens(train_data, SRC_LANGUAGE))
TRG_VOCAB = build_vocab_from_iterator(yield_tokens(train_data, TRG_LANGUAGE))

# Insert special tokens manually
for tok in reversed(special_symbols):
    SRC_VOCAB.insert_token(tok, 0)
    TRG_VOCAB.insert_token(tok, 0)

SRC_VOCAB.set_default_index(UNK_IDX)
TRG_VOCAB.set_default_index(UNK_IDX)

# --------------------------------------------------
# Model hyperparameters (MUST match training)
# --------------------------------------------------
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TRG_VOCAB)
HID_DIM = 256
N_LAYERS = 3
N_HEADS = 8
PF_DIM = 512
DROPOUT = 0.1

# --------------------------------------------------
# Load model
# --------------------------------------------------
encoder = Encoder(
    INPUT_DIM, HID_DIM, N_LAYERS, N_HEADS,
    PF_DIM, DROPOUT, device
)

decoder = Decoder(
    OUTPUT_DIM, HID_DIM, N_LAYERS, N_HEADS,
    PF_DIM, DROPOUT, device
)

model = Seq2SeqTransformer(
    encoder, decoder,
    PAD_IDX, PAD_IDX,
    device
).to(device)

model.load_state_dict(
    torch.load("transformer_general.pt", map_location=device)
)
model.eval()

# --------------------------------------------------
# Translation function
# --------------------------------------------------
def translate_sentence(sentence, max_len=50):
    tokens = tokenize(sentence)
    src_indexes = [BOS_IDX] + [SRC_VOCAB[t] for t in tokens] + [EOS_IDX]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [BOS_IDX]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            trg_mask = model.make_trg_mask(trg_tensor)
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break

    trg_tokens = [TRG_VOCAB.lookup_token(i) for i in trg_indexes]

    return " ".join(trg_tokens[1:-1])

# --------------------------------------------------
# Dash App
# --------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"width": "60%", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H1("English → Sinhala Translation"),

        dcc.Textarea(
            id="input-text",
            placeholder="Enter English sentence...",
            style={"width": "100%", "height": 100}
        ),

        html.Br(),

        html.Button("Translate", id="translate-btn"),

        html.Hr(),

        html.H3("Sinhala Translation"),
        html.Div(id="output-text", style={"fontSize": 18, "color": "darkblue"})
    ]
)

@app.callback(
    Output("output-text", "children"),
    Input("translate-btn", "n_clicks"),
    Input("input-text", "value")
)
def translate_callback(n_clicks, text):
    if not n_clicks or not text:
        return ""
    return translate_sentence(text)

# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
