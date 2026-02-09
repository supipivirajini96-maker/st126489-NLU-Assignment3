# ===============================
# Imports
# ===============================
import pickle
import torch
import nltk
from nltk.tokenize import word_tokenize

import dash
from dash import html, dcc
from dash.dependencies import Input, Output

from model import Encoder, Decoder, Seq2SeqTransformer

# ===============================
# NLTK
# ===============================
nltk.download("punkt")

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load vocabularies
# ===============================
with open("saved_vocab/src_stoi.pkl", "rb") as f:
    SRC_STOI = pickle.load(f)

with open("saved_vocab/tgt_stoi.pkl", "rb") as f:
    TGT_STOI = pickle.load(f)

with open("saved_vocab/tgt_itos.pkl", "rb") as f:
    TGT_ITOS = pickle.load(f)

# ===============================
# Auto-detect special tokens
# ===============================
def find_token(vocab, candidates):
    for t in candidates:
        if t in vocab:
            return t
    raise ValueError(f"Missing token from {candidates}")

SOS_TOKEN = find_token(TGT_STOI, ["<sos>", "<bos>", "sos", "bos"])
EOS_TOKEN = find_token(TGT_STOI, ["<eos>", "</s>", "eos"])
PAD_TOKEN = find_token(TGT_STOI, ["<pad>", "pad"])
UNK_TOKEN = find_token(TGT_STOI, ["<unk>", "unk"])

SOS_IDX = TGT_STOI[SOS_TOKEN]
EOS_IDX = TGT_STOI[EOS_TOKEN]
PAD_IDX = TGT_STOI[PAD_TOKEN]
UNK_IDX = TGT_STOI[UNK_TOKEN]

print("✅ Tokens:", SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN)

# ===============================
# Model hyperparameters
# ===============================
INPUT_DIM = len(SRC_STOI)
OUTPUT_DIM = len(TGT_STOI)

HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# ===============================
# Build model
# ===============================
enc = Encoder(
    INPUT_DIM,
    HID_DIM,
    ENC_LAYERS,
    ENC_HEADS,
    ENC_PF_DIM,
    ENC_DROPOUT,
    device
)

dec = Decoder(
    OUTPUT_DIM,
    HID_DIM,
    DEC_LAYERS,
    DEC_HEADS,
    DEC_PF_DIM,
    DEC_DROPOUT,
    device
)

MODEL = Seq2SeqTransformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
MODEL.load_state_dict(torch.load("transformer_general.pt", map_location=device))
MODEL.eval()

print("✅ Model loaded successfully (GENERAL attention)")

# ===============================
# Translation function
# ===============================
def translate_sentence(sentence, max_len=50):
    tokens = [SOS_TOKEN] + word_tokenize(sentence.lower()) + [EOS_TOKEN]

    src_indexes = [SRC_STOI.get(t, SRC_STOI.get(UNK_TOKEN)) for t in tokens]


    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = MODEL.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = MODEL.encoder(src_tensor, src_mask)

    trg_indexes = [SOS_IDX]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = MODEL.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, _ = MODEL.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(-1)[:, -1].item()

        # stop repetition
        if len(trg_indexes) >= 3:
           if trg_indexes[-2:] == [pred_token, pred_token]:
              break


        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break

    trg_tokens = [
        TGT_ITOS[i]
        for i in trg_indexes
        if i not in {SOS_IDX, EOS_IDX, PAD_IDX}
    ]

    return " ".join(trg_tokens)

# ===============================
# Dash App
# ===============================
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"width": "60%", "margin": "auto", "fontFamily": "Arial"},
    children=[
        html.H2("English → Sinhala Translator"),

        dcc.Textarea(
            id="input-text",
            placeholder="Enter English sentence...",
            style={"width": "100%", "height": 100},
        ),

        html.Br(),
        html.Button("Translate", id="translate-btn"),

        html.Hr(),

        html.Div(
            id="output-text",
            style={
                "whiteSpace": "pre-wrap",
                "fontSize": 20,
                "color": "darkgreen",
            },
        ),
    ],
)

# ===============================
# Callback
# ===============================
@app.callback(
    Output("output-text", "children"),
    Input("translate-btn", "n_clicks"),
    Input("input-text", "value"),
)
def translate_callback(n_clicks, text):
    if not n_clicks or not text:
        return ""
    return translate_sentence(text)

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    app.run(debug=False)
