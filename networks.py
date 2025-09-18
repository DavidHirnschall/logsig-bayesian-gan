
from utils import *


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()#LeakyReLU()#nn.ReLU()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y

class ResFNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], flatten: bool = False, emb=False):
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.linear_sig = nn.Linear(140, 140)
        self.network = nn.Sequential(*blocks)
        self.network
        self.blocks = blocks
        self.emb = emb
        if self.emb:
            self.emb1 = nn.Embedding(5,5) #category
            self.emb2 = nn.Embedding(8,8) #age
            self.emb3 = nn.Embedding(2,2) #gender
            self.dropout_emb = nn.Dropout(p=0.2)
            self.activation = nn.Tanh()#nn.LeakyReLU()
            
    def forward(self, x, embeddings=None):
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        
        if self.emb and embeddings != None:
            emb_cat = self.activation(self.emb1(embeddings[:,2]))
            emb_age = self.activation(self.emb2(embeddings[:,0]))
            emb_gen = self.activation(self.emb3(embeddings[:,1]))
            x = self.activation(self.linear_sig(x))
            x = torch.cat([x, emb_cat, emb_age, emb_gen],dim=1).float()
            x = self.dropout_emb(x)
        
        out = self.network(x)
        return out

class ResFNNDiscriminator(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: Tuple[int], output_dim=3, emb=False):
        super(ResFNNDiscriminator, self).__init__()

        self.resfnn = ResFNN(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, emb=emb)
        self.resfnn.apply(init_weights_d(gain=1))

    def forward(self, x: torch.Tensor, embeddings=None):
        batch_size = x.shape[0]
        return self.resfnn(x.reshape(batch_size, -1), embeddings)


class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward_(self, batch_size: int, n_lags: int, device: str):
        """ Implement here generation scheme. """
        #...
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x


class ResFNNGenerator(GeneratorBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: int, n_layers: int, init_fixed: bool = True, len_noise=256):
        super(ResFNNGenerator, self).__init__(input_dim, output_dim)
        self.len_noise = len_noise
        blocks = list()
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim[0]))
        self.seq_nn = nn.Sequential(*blocks) 
        self.output_dim = output_dim
        self.linear_sig = nn.Linear(input_dim_block, output_dim[0])
        self.linear_sig.apply(init_weights)
        self.emb1 = nn.Embedding(5,5) #category
        self.emb2 = nn.Embedding(8,8) #age
        self.emb3 = nn.Embedding(2,2) #gender
        self.activation = nn.Tanh()

    def forward(self, batch_size: int, n_lags: int, device: str, z=None, embeddings=None) -> torch.Tensor:
        if z == None:
            z = (0.1 * torch.randn(batch_size, self.len_noise)).to(device)
        z = z

        if embeddings != None:
            emb_cat = self.activation(self.emb1(embeddings[:,2]))
            emb_age = self.activation(self.emb2(embeddings[:,0]))
            emb_gen = self.activation(self.emb3(embeddings[:,1]))
            #emb_mer = self.activation(self.emb4(embeddings[:,3]))
            z = torch.cat([z, emb_cat, emb_age, emb_gen],dim=1).float() #, emb_mer
        
        x = self.seq_nn(z)
        return x


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.0, bidirectional=False):
        super(GRUClassifier, self).__init__()
        # dim: batch x len x feature
        self.gru = nn.GRU(
            input_size=input_dim,      # here: 4 features
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len=5, features=4]
        out, h_n = self.gru(x)  
        # h_n: [num_layers * num_directions, batch, hidden_dim]

        # take last hidden state
        last_hidden = h_n[-1]  # [batch, hidden_dim * num_directions]
        logits = self.fc(last_hidden)  # [batch, output_dim]
        return logits