
import torch
import torch.nn as nn

class Highway(nn.Module):
    
    def __init__(self,embed_size):
        super(Highway,self).__init__()
        self.projection = nn.Linear(embed_size,embed_size,bias=True)
        self.gate = nn.Linear(embed_size,embed_size,bias=True)
        self.Relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
    def forward(self,x_conv_out):
        proj_out = self.projection(x_conv_out)
        proj_relu = self.Relu(proj_out)

        gate_out = self.gate(proj_relu)
        gate_sigmoid = self.sigmoid(gate_out)

        highway_out = torch.mul(proj_relu,gate_sigmoid) + torch.mul(x_conv_out,(1-gate_sigmoid))
        x_word_embed = self.dropout(highway_out)
        
        return x_word_embed

#     ###### Test
# if __name__ == "__main__":
#     batch_size = 2
#     embed_size = 6
#     model = Highway(embed_size)
#     criterion = nn.CrossEntropyLoss()
#     t3 = torch.randn(batch_size, embed_size)
    
#     output = model(t3)
#     print(output)