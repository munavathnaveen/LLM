import torch 
import torch.nn as nn
import tiktoken 
from torch.utils.data import Dataset,DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.44715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_sample(model, idx, max_new_tokens, context_size):
    for i in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx 

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
text="Every Naveen "
encoded=tokenizer.encode(text)
encoded_tensor=torch.tensor(encoded).unsqueeze(0)

model.eval()
out=generate_text_sample(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=1,
    context_size=GPT_CONFIG_124M["context_length"]  
)
# print(out)

decode_text=tokenizer.decode(out.squeeze(0).tolist())

inputs = torch.tensor([[16833, 3626, 6100],   
                       [40,    1107, 588]])   

targets = torch.tensor([[3626, 6100, 345  ],  
                        [1107,  588, 11311]]) 

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1) 
print(probas.shape) 

logits_flat=logits.flatten(0,1)
target_flat=targets.flatten()

loss=torch.nn.functional.cross_entropy(logits_flat,target_flat)

print(loss)

# perplexity=torch.exp(loss)

# print(perplexity)


# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# # print(raw_text[:99])
# total_character=len(raw_text)

# total_tokens=len(tokenizer.encode(raw_text))

# print(total_character)
# print(total_tokens)

# class GPTDatasetV1(Dataset):
#     def __init__(self,text,tokenizer,max_length,stride):
#         self.input_ids=[]
#         self.target_ids=[]
#         token_ids=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
#         for i in range(0,len(token_ids)-max_length,stride):
#             input_chunk=token_ids[i:i+max_length]
#             target_chunk=token_ids[i+1:max_length+1]
#             self.input_ids.append(input_chunk)
#             self.target_ids.append(target_chunk)
#     def __len__(self):
#         return len(self.input_ids)
#     def __getitem__(self, idx):
#         return self.input_ids[idx],self.target_ids[idx]
# def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
#     tokenizer=tiktoken.get_encoding("gpt2")
#     dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)
#     dataloader=DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,
#         num_workers=num_workers
#     ) 
#     return dataloader

# train_ratio=0.9
# split_ids=int(train_ratio*len(raw_text))
# train_data=raw_text[:split_ids]
# val_data=raw_text[split_ids:]
# print(val_data)