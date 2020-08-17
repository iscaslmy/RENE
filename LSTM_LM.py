import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# RNN语言模型
class LSTM_LM(nn.Module):  # RNNLM类继承nn.Module类
    def __init__(self, vocab_size,
                 embedding_dim=100, hidden_dim=128, batch_size=20, dropout=0.5):
        super(LSTM_LM, self).__init__()

        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # 嵌入层 one-hot形式(vocab_size,1) -> (embed_size,1)
        # word embedding layer
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

        # bilstm layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers=1, bidirectional=False,
                            batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # 输出层的全联接操作
        self.linear = nn.Linear(self.hidden_dim, self.vocab_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, self.batch_size, self.hidden_dim).to(device),
                torch.randn(1, self.batch_size, self.hidden_dim).to(device))

    def forward(self, x):
        self.hidden = self.init_hidden()
        # sentence_length = x.shape[1]
        # 词嵌入
        x = self.word_embeds(x)
        # embeddings = self.word_embeds(x).view(self.batch_size, sentence_length, -1)

        # LSTM前向运算
        ltsm_out, self.hidden = self.lstm(x, self.hidden)

        # 每个时间步骤上LSTM单元都会有一个输出，batch_size个样本并行计算(每个样本/序列长度一致)
        # out (batch_size,sequence_length,hidden_size)
        # 把LSTM的输出结果变更为(batch_size*sequence_length, hidden_size)的维度
        ltsm_out = ltsm_out.reshape(ltsm_out.size(0) * ltsm_out.size(1), ltsm_out.size(2))
        ltsm_out = self.dropout(ltsm_out)
        # 全连接 (batch_size*sequence_length, hidden_size)->(batch_size*sequence_length, vacab_size)
        out = self.linear(ltsm_out)

        return out

