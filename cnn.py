import torch
import torch.nn as nn

class CNN(nn.Module):
    
    def __init__(self,embed_size,feature_size,kernel_size,max_word_length):
        super(CNN,self).__init__()
        self.conv = nn.Conv1d(in_channels=embed_size,out_channels=feature_size,kernel_size=kernel_size,bias=True)
        self.Relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(max_word_length-kernel_size+1)

    def forward(self,x_reshaped):
        conv_out = self.conv(x_reshaped)
        Relu_out = self.Relu(conv_out)
        pool_out = self.max_pool(Relu_out)

        return pool_out.squeeze(2)
##### TEST
# if __name__ =='__main__':
#     embed_size = 16
#     kernel_size = 5
#     max_word_length = 21
#     batch_size = 2 
#     features = 16
#     model = CNN(embed_size,features,kernel_size,max_word_length)

#     t3 = torch.randn(batch_size, embed_size,max_word_length)

#     output = model(t3)
#     print(output.shape)
#     output = output.squeeze(2)
#     print(output)

