# Neural network model to generate embeddings
class BeerAttributeNN(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(BeerAttributeNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)  # Embedding dimension of 32

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x