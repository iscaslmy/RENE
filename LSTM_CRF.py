import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1).to(device)


class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix,
                 embedding_dim=100, hidden_dim=128, batch_size=20, dropout=0.5):
        super(LSTM_CRF, self).__init__()

        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tag_size = len(tag_to_ix)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size

        # word embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # bilstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, bidirectional=False,
                            batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, self.batch_size, self.hidden_dim).to(device),
                torch.randn(1, self.batch_size, self.hidden_dim).to(device))

    def __get_lstm_features(self, sentences):
        # batched
        self.hidden = self.init_hidden()

        sentence_length = sentences.shape[1]

        embeddings = self.word_embeds(sentences).view(self.batch_size, sentence_length, -1)

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, logits, label):
        '''
        caculate real path score
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]

        Score = Emission_Score + Transition_Score
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])
        '''
        score = torch.zeros(1).to(device)
        label = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), label])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag_to_ix[STOP_TAG]]
        return score

    def _forward_alg(self, logits):
        """
        caculate total score

        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]

        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        obs = []
        previous = torch.full((1, self.tag_size), 0).to(device)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_to_ix[STOP_TAG]]
        # caculate total_scores
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores.to(device)

    def neg_log_likelihood(self, sentences, tags, lengths):
        # batched
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        real_path_score = torch.zeros(1).to(device)
        total_score = torch.zeros(1).to(device)
        for logit, tag, leng in zip(logits, tags, lengths):
            logit = logit[:leng]
            tag = tag[:leng]
            real_path_score += self._score_sentence(logit, tag)
            total_score += self._forward_alg(logit)
        # print("total score ", total_score)
        # print("real score ", real_path_score)
        return total_score - real_path_score

    def forward(self, sentences, lengths=None):  # dont confuse this with _forward_alg above.
        """
        :param sentences: sentences to predict
        :param lengths: lengths represent the ture length of sentence, the default is sentences.size(-1)
        :return:
        """
        sentences = torch.tensor(sentences, dtype=torch.long).to(device)
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]
            score, path = self.__viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def __viterbi_decode(self, logits):
        trellis = torch.zeros(logits.size()).to(device)
        backpointers = torch.zeros(logits.size(), dtype=torch.long).to(device)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.cpu().numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi




