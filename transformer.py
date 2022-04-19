import torch
from torch.nn import Module, Linear, MSELoss, ModuleList, Conv1d, Dropout, LayerNorm, parameter
import torch.nn.functional as F


def do_attention(query,key,value):
    # get scaled attention scores
    attention_scores = torch.bmm(query, key.transpose(1,2))/torch.sqrt(query.size(-1))
    attention_weights = F.softmax(attention_scores,dim=1)
    return torch.bmm(attention_weights, value)    




class AttentionHead(Module):
    
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.Wq = Linear(embed_dim, head_dim)
        self.Wk = Linear(embed_dim, head_dim)
        self.Wv = Linear(embed_dim, head_dim)
    
    def forward(self, h):
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        outputs = do_attention(q,k,h)
        
        
        
        
class MultiHeadAttention(Module):
    
    def __init__(self, hidden_size, num_heads) -> None:
        super().__init__()
        num_heads = num_heads
        # by convention
        head_dim = hidden_size // num_heads
        self.heads = ModuleList(
            [AttentionHead(hidden_size, head_dim) for _ in range(num_heads)]
        )
        self.output = Linear(hidden_size, hidden_size)
        
    def forward(self, h):
        x = torch.cat([head(h) for head in self.heads], dim = 1)
        return self.output(x) 


class Sine(Module):
    
    def __init__(self,input_size, output_size) -> None:
        
        super().__init__()
        self.out = output_size
        self.weights_linear = parameter.Parameter(torch.randn(input_size, 1))
        self.bias_linear = parameter.Parameter(torch.randn(1))
        self.weights_periodic = parameter.Parameter(torch.randn(in_features, output_size-1))
        self.bias_periodic = parameter.Parameter(torch.randn(output_size-1))
    
    def forward(self, x):
        x = torch.mean(x, axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = torch.expand_dims(time_linear, axis=-1) 
    
        time_periodic = torch.sin(torch.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = torch.expand_dims(time_periodic, axis=-1) 
        return torch.concat([time_linear, time_periodic], axis=-1)
    
    
    
    
class FeedForward(Module):
    
    # rule of thumb: hidden size of first layer 4x emebddding dimension
     def __init__(self,hidden_size, intermediate_size, dropout_prob = 0.3) -> None:
        super().__init__()
        # equivalent to dense layer or a position wise feed-forward network
        self.conv1 = Conv1d(in_channels = hidden_size, out_channels = intermediate_size, kernel_size=1)
        self.conv2 = Conv1d(in_channels = intermediate_size, out_channels = hidden_size, kernel_size = 1)
        # standard to use gelu
        self.gelu = F.GELU()
        self.dropout = Dropout(dropout_prob)
    
     def forward(self,x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return self.dropout(x)
    
    
    
class TransformerEncoderLayer(Module):
    
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout_prob) -> None:
        super().__init__()
        # layer norm is prefered for transformer
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.ff = FeedForward(hidden_size, intermediate_size, dropout_prob)
        
    
    def forward(self, x):
        hidden = self.layer_norm1(x)
        #skip connection as in resnet
        x = x + self.attention(hidden)
        x = self.layer_norm2(x)
        # skip connection
        return x + self.ff(x)


class Time2Vec(Module):
    
    def __init__(self, seq_len) -> None:
        super().__init__()
        self.periodic =  Sine(1, seq_len)
        self.linear = Linear(seq_len,2)
        
    def forward(self, x):
        x = self.periodic(x)
        return self.linear(x)
    
    
    
class TransformerEncoder(Module):
    
    def __init__(self, num_hidden, hidden_size, intermediate_size, num_heads, dropout_prob, seq_len) -> None:
        super().__init__()
        self.time_embedding = Time2Vec(seq_len)
        self.layers = ModuleList([TransformerEncoderLayer(hidden_size, intermediate_size, num_heads, dropout_prob)
                                 for _ in range(num_hidden)])
        
    def forward(self, x):
        # get time embedding
        time = self.time_embedding(x)
        # cncatenate it to input
        x = torch.concat([x,time],axis=-1)
        for layer in self.layers:
            x = layer(x)
        return x
    