import torch
import timm
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, AutoTokenizer

class QuickGELU(torch.nn.Module):
   def forward(self, x: torch.Tensor):
       return x * torch.sigmoid(1.702 * x)
   
class CLIPAttentiveLayer(nn.Module):
    def __init__(self, n_head, d_embed):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_embed)
        # self.attention = SelfAttention(n_head, d_embed)
        self.attention = nn.MultiheadAttention(d_embed, n_head, batch_first=True)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)
        self.activation = QuickGELU()

    def forward(self, x, causal_mask):
        residue = x
        x = self.layernorm_1(x)
        # x = self.attention(x, causal_mask = True)
        x, _ = self.attention(x, x, x, is_causal = True, attn_mask = causal_mask) if causal_mask is not None else self.attention(x, x, x)
        x += residue
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x += residue
        return x
   
class CLIPProjectionHead(nn.Module):
    def __init__(self, args, head_type="image"):
        super().__init__()
        if head_type == "image":
            input_dim = args.image_embed_dim
        elif head_type == "text":
            input_dim = args.text_embed_dim
        self.projection = nn.Linear(input_dim, args.context_dim)
        self.gelu = QuickGELU()
        self.fc = nn.Linear(args.context_dim, args.context_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.context_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        x = self.layer_norm(x)
        return x   
    
class CLIPImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = timm.create_model(args.image_encoder_model_name, 
                                       pretrained=args.pretrained_image_encoder_backbone, num_classes=0, global_pool="avg") 
        self.attention_layers = nn.ModuleList([
            CLIPAttentiveLayer(args.num_heads, args.image_embed_dim) 
            for i in range(args.num_layers_image_encoder)])
        self.projection_head = CLIPProjectionHead(args, head_type="image")
        # self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear((args.image_size // args.image_backbone_stride_factor)**2, 1)
        self.device = args.device
        for p in self.model.parameters():
            p.requires_grad = args.train_image_encoder_backbone

    def forward(self, image):
        state = self.model.forward_features(image)
        grid_size = state.shape[2]
        channels = state.shape[1]
        batch = state.shape[0]
        state = state.view(batch, channels, grid_size * grid_size).swapaxes(1, 2)
        #causal_mask = torch.triu(torch.ones(state.shape[1], state.shape[1]), diagonal=1).bool().to(self.device)
        for layer in self.attention_layers:
            state = layer(state, None)   # causal mask not needed for image features
        out = self.projection_head(state)
        weighted_vector = self.linear(state.swapaxes(2,1)).squeeze(-1)
        weighted_vector = self.projection_head(weighted_vector)
        # out = torch.nn.functional.normalize(out, p=2.0, dim=-1)
        # weighted_vector = torch.nn.functional.normalize(weighted_vector, p=2.0, dim=-1)
        return out, weighted_vector

class CLIPTextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        if not args.pretrained_text_encoder_backbone:
            self.model = DistilBertModel.from_pretrained(args.text_encoder_model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
        self.attention_layers = nn.ModuleList([
            CLIPAttentiveLayer(args.num_heads, args.text_embed_dim) 
            for i in range(args.num_layers_text_encoder)])
        self.projection_head = CLIPProjectionHead(args, head_type="text")
        self.device = args.device
        self.linear = nn.Linear(args.num_tokens, 1)
        for p in self.model.parameters():
            p.requires_grad = args.train_text_encoder_backbone

    def forward(self, input_ids, attention_masks):
        state = self.model(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state
        # causal_mask = torch.triu(torch.ones(state.shape[1], state.shape[1]), diagonal=1).bool().to(self.device)
        for layer in self.attention_layers:
            state = layer(state, None) # causal mask not needed, let text feature attend to all other features in caption
        out = self.projection_head(state)
        weighted_vector = self.linear(state.swapaxes(2,1)).squeeze(-1)
        weighted_vector = self.projection_head(weighted_vector)
        # out = torch.nn.functional.normalize(out, p=2.0, dim=-1)
        # weighted_vector = torch.nn.functional.normalize(weighted_vector, p=2.0, dim=-1)
        return out, weighted_vector

class CLIPModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_tokens = args.num_tokens
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model_name)
        self.image_encoder = CLIPImageEncoder(args)
        self.text_encoder = CLIPTextEncoder(args)
        self.target_token_idx = 0

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, images, input_ids, attention_masks):
        _, image_embed = self.image_encoder(images)
        _, text_embed = self.text_encoder(input_ids, attention_masks)
        # target_text_embed = self.text_encoder(input_ids, attention_masks)[0][:, self.target_token_idx, :]
        return image_embed, text_embed
    
    def query(self, images, texts):
        encoded_query = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.num_tokens)
        batch = {k: torch.tensor(v).to(self.device) for k,v in encoded_query.items()}
        return self.forward(images, batch["input_ids"], batch["attention_mask"])

    def encode_texts(self, texts):
        encoded_query = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.num_tokens)
        batch = {k: torch.tensor(v).to(self.device) for k,v in encoded_query.items()}
        return self.text_encoder(batch["input_ids"], batch["attention_mask"])
    
    def encode_images(self, images):
        _, image_embed = self.image_encoder(images)
        return image_embed

    def find_matches(self, query_texts, truth_image_embeddings, max_matches=5, num_captions=1):
        
        if type(query_texts) == type(str):
            query_texts = [query_texts]

        encoded_query_texts = self.tokenizer(query_texts, padding='max_length', truncation=True, max_length=self.num_tokens)
        batch = {k: torch.tensor(v).to(self.device) for k,v in encoded_query_texts.items()}
        
        _, text_embeddings = self.text_encoder(batch['input_ids'], batch['attention_mask'])
        # text_embeddings = self.text_encoder(batch['input_ids'], batch['attention_mask'])[0][:, self.target_token_idx, :]

        norm_truth_image_embeddings = F.normalize(truth_image_embeddings, p=2, dim=-1)
        norm_text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        similarity = norm_text_embeddings @ norm_truth_image_embeddings.T
        
        values, indices = torch.topk(similarity, max_matches * num_captions, dim=-1)
        indices = indices.cpu().numpy()

        match_indices = []
        for idx in indices:
            match_indices.append(idx[:max_matches])

        return match_indices

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, d_embed, n_token):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, d_embed)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, d_embed)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x 
    
class CLIPAttentiveTextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = CLIPEmbedding(args.vocab_size, 
                                       args.text_embed_dim, 
                                       args.num_tokens)
        
        self.layers = nn.ModuleList([
            CLIPAttentiveLayer(args.num_heads, args.text_embed_dim) 
            for i in range(args.num_layers)])
        
        self.layernorm = nn.LayerNorm(args.text_embed_dim)

    def forward(self, tokens):
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity):
    text_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (text_loss + image_loss) / 2.0

def clip_metrics(similarity):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc