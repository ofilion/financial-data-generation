import torch
from torch.nn import Module, Linear, MSELoss, ModuleList, Conv1d, Dropout, LayerNorm, parameter, GELU
import torch.nn.functional as F
import math


def do_attention(query,key,value):
    # get scaled attention scores
    attention_scores = torch.bmm(query, key.transpose(1,2))/math.sqrt(query.size(-1))
    attention_weights = F.softmax(attention_scores,dim=1)
    return torch.bmm(attention_weights, value)    


class AttentionHead(Module):
    
    def __init__(self, embed_dim, head_dim) -> None:
        super().__init__()
        self.Wq = Linear(embed_dim, head_dim)
        self.Wk = Linear(embed_dim, head_dim)
        self.Wv = Linear(embed_dim, head_dim)
    
    def forward(self, h):
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        outputs = do_attention(q,k,v)
        return outputs
        
        
class MultiHeadAttention(Module):
    
    def __init__(self, hidden_size, num_heads) -> None:
        super().__init__()
        num_heads = num_heads
        # by convention
        head_dim = hidden_size // num_heads
        self.heads = ModuleList(
            [AttentionHead(hidden_size, head_dim) for _ in range(num_heads)]
        )
        
      #  self.output = Linear(hidden_size*num_heads, hidden_size)
        self.output = Linear(hidden_size, hidden_size)
        
    def forward(self, h):
        x = torch.cat([head(h) for head in self.heads], dim = -1)
        return self.output(x) 


class Sine(Module):
    
    def __init__(self,input_size) -> None:
        
        super().__init__()
        self.weights_linear = parameter.Parameter(torch.randn(input_size))
        self.bias_linear = parameter.Parameter(torch.randn(input_size))
        self.weights_periodic = parameter.Parameter(torch.randn(input_size))
        self.bias_periodic = parameter.Parameter(torch.randn(input_size))
    
    def forward(self, x):
     #   print(x.shape, self.weights_linear.shape)
        time_linear = x * self.weights_linear + self.bias_linear

        time_linear = torch.unsqueeze(time_linear,-1)
      
        
        time_periodic = torch.sin( x * self.weights_periodic + self.bias_periodic)
        time_periodic = torch.unsqueeze(time_periodic,-1)
        
        return torch.cat([time_linear, time_periodic], -1)
    
    
class Time2Vec(Module):
    
    def __init__(self, seq_len) -> None:
        super().__init__()
        self.periodic =  Sine(seq_len)
        
    def forward(self, x):
        x = torch.mean(x, axis=-1)
        x = self.periodic(x)
        return x
    
    
class FeedForward(Module):
    
    # rule of thumb: hidden size of first layer 4x emebddding dimension
     def __init__(self,hidden_size, inter_size, dropout_prob = 0.3) -> None:
        super().__init__()
        # equivalent to dense layer or a position wise feed-forward network
       # self.conv1 = Conv1d(in_channels = seq_len, out_channels = hidden_dim, kernel_size=1)
       # self.conv2 = Conv1d(in_channels = hidden_dim, out_channels = seq_len, kernel_size = 1)
        self.conv1 = Linear(hidden_size, inter_size)
        self.conv2 = Linear(inter_size, hidden_size)
        # standard to use gelu
        self.gelu = GELU()
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
        self.ff = FeedForward(hidden_size ,intermediate_size, dropout_prob)
        
    
    def forward(self, x):
        hidden = self.layer_norm1(x)
        #skip connection as in resnet
        x = x + self.attention(hidden)
        x = self.layer_norm2(x)
        # skip connection
        return x + self.ff(x)

    
class TransformerEncoder(Module):
    
    def __init__(self, num_hidden, hidden_size, intermediate_size, 
                         num_heads, dropout_prob, seq_len) -> None:
        
        super().__init__()
        self.hidden_dim = hidden_size
        self.time_embedding = Time2Vec(seq_len)
        self.layers = ModuleList([TransformerEncoderLayer(hidden_size, intermediate_size, num_heads, dropout_prob)
                                 for _ in range(num_hidden)])
        
    def forward(self, x):
        # get time embedding
        time = self.time_embedding(x)
        # concatenate it to input
        x = torch.cat([x,time],axis=-1)
        for layer in self.layers:
            x = layer(x)
        return x
    

class TransformerForPrediction(Module):
    
    def __init__(self, encoder: TransformerEncoder, dropout_prob = 0.3) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = Dropout(dropout_prob)
        self.fc = Linear(encoder.hidden_dim,1)
        
    def forward(self, x):
       
        x = self.encoder(x)
        x = self.dropout(x)
        ## -3 because we added 2 dimensions for time at the end, assuming price isn the last element
        ## might be worth adding this as an argument to prevent confusion
        x = self.fc(x)[:,-3,:]
        return x
        
        
class TransformerForBinaryClassification(Module):
    
    def __init__(self,encoder: TransformerEncoder, dropout_prob = 0.3) -> None:
        
        self.encoder = encoder
        self.dropout = Dropout(dropout_prob)
        self.l1 = Linear(encoder.hidden_dim,encoder.hidden_dim)
        self.out = Linear(encoder.hidden_dim,1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.l1(x)
        x = GELU(x)
        return self.out(x)
        