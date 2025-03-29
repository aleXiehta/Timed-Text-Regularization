import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchaudio.functional import resample
from transformers import AutoProcessor, Wav2Vec2Model, HubertModel

class SSLPreTrained(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ssl_pretrained = WavLMModel.from_pretrained("microsoft/wavlm-base")
    
    def forward(self, x):
        return self.ssl_pretrained(x)
    
class Summarizer(pl.LightningModule):
    def __init__(self, num_layers=4, d_model=768):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.translator = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.inter = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.ssl_pretrained = SSLPreTrained()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        
    def get_acoustic_features(self, audio):
        with torch.no_grad():
            B = audio.size(0)
            out = self.ssl_pretrained(resample(audio, orig_freq=8000, new_freq=16000).view(B, -1))
        return out.last_hidden_state
        
    def sentence2subwords(self, acoustic_features, repeat_ratios):
        N = repeat_ratios.size(0)
        num_frames = acoustic_features.size(1)

        # # get sequence lengths
        boundaries = torch.cat(
            [torch.zeros(N, 1).to(acoustic_features.device), repeat_ratios.cumsum(1) * num_frames],
            dim=1
        ) # word boundaries in idx
        boundaries = torch.round(boundaries)
        lengths = boundaries.diff(1)
        if any(lengths.view(-1) < 0):
            print(lengths)
        # split padded sequences by lengths and remove zero-length sequences
        splitted = torch.split(acoustic_features.view(-1, self.d_model), lengths.view(-1).int().tolist())
        zero_indices = (lengths.view(-1) == 0).nonzero().view(-1).tolist()
        splitted = list(splitted)
        for zid in sorted(zero_indices, reverse=True):
            del splitted[zid]
        splitted = nn.utils.rnn.pad_sequence(splitted, batch_first=True)
        return splitted, lengths

    def run_and_match(self, auc_f, repeat_ratios, subword_f):
        sub_auc_f, subword_lengths = self.sentence2subwords(auc_f, repeat_ratios)
        sub_auc_f = torch.cat([self.cls.repeat(sub_auc_f.shape[0], 1, 1), sub_auc_f], dim=1)
        auc_summary = self.translator(sub_auc_f)[:, 0, :]
        subword_f_expanded = []
        start = 0
        for i, (swf, r) in enumerate(zip(subword_f.view(-1, self.d_model), subword_lengths.view(-1))):
            r = int(r.item())
            if r > 0:
                subword_f_expanded.append(swf)
        subword_f_expanded = torch.stack(subword_f_expanded, dim=0)
        subword_lengths = subword_lengths.view(-1)
        subword_lengths = subword_lengths[subword_lengths.nonzero()]
        auc_summary = self.inter(auc_summary.unsqueeze(0)).squeeze(0)
        return auc_summary, subword_f_expanded
    
    def pairwise_cosine_similarity(self, x, y):
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
        return x @ y.T
        
    def criterion(self, auc_cls, subword_f_expanded, tau=0.5):
        cos_dist = 1 - F.cosine_similarity(auc_cls, subword_f_expanded).mean()
        pw_cos_sim_auc = self.pairwise_cosine_similarity(auc_cls, auc_cls) / tau
        pw_cos_sim_sub = self.pairwise_cosine_similarity(subword_f_expanded, subword_f_expanded) / tau
        loss = cos_dist + F.mse_loss(pw_cos_sim_auc, pw_cos_sim_sub)
        return loss

    def training_step(self, batch, batch_idx):
        m, s, [f1, f2], [r1, r2] = batch
        auc_f1 = self.get_acoustic_features(s[:, 0, :])
        auc_f2 = self.get_acoustic_features(s[:, 1, :])
        auc_cls1, subword_f1_expanded = self.run_and_match(auc_f1, r1, f1)
        auc_cls2, subword_f2_expanded = self.run_and_match(auc_f2, r2, f2)
        loss = self.criterion(torch.cat([auc_cls1, auc_cls2], dim=0), torch.cat([subword_f1_expanded, subword_f2_expanded], dim=0), tau=0.5)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        m, s, [f1, f2], [r1, r2] = batch
        auc_f1 = self.get_acoustic_features(s[:, 0, :])
        auc_f2 = self.get_acoustic_features(s[:, 1, :])
        auc_cls1, subword_f1_expanded = self.run_and_match(auc_f1, r1, f1)
        auc_cls2, subword_f2_expanded = self.run_and_match(auc_f2, r2, f2)
        loss = self.criterion(torch.cat([auc_cls1, auc_cls2], dim=0), torch.cat([subword_f1_expanded, subword_f2_expanded], dim=0), tau=0.5)
        self.log("valid_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-4, betas=(0.9, 0.98))
        return optimizer
    
    def forward(self, hubert_embeddings):
        return self.translator(hubert_embeddings)
    
def rnn_collate(batch, n_src):
    m = torch.stack([b[0] for b in batch], dim=0)
    s = torch.stack([b[1] for b in batch], dim=0)
    
    if n_src == 3:
        f1 = [b[2] for b in batch]
        f2 = [b[3] for b in batch]
        f3 = [b[4] for b in batch]
        r1 = [b[5] for b in batch]
        r2 = [b[6] for b in batch]
        r3 = [b[7] for b in batch]
        [f1, f2, f3] = map(partial(lambda x: pad_sequence(x, batch_first=True)), [f1, f2, f3])
        [r1, r2, r3] = map(partial(lambda x: pad_sequence(x, batch_first=True)), [r1, r2, r3])
        return m, s, [f1, f2, f3], [r1, r2, r3]
    else:
        f1 = [b[2] for b in batch]
        f2 = [b[3] for b in batch]
        r1 = [b[4] for b in batch]
        r2 = [b[5] for b in batch]

        [f1, f2] = map(partial(lambda x: pad_sequence(x, batch_first=True)), [f1, f2])
        [r1, r2] = map(partial(lambda x: pad_sequence(x, batch_first=True)), [r1, r2])
        # import pdb; pdb.set_trace()
        return m, s, [f1, f2], [r1, r2]
    
class LMLoss(pl.LightningModule):
    def __init__(
        self, 
        translator_path=None,
    ):
        super().__init__()
        self.translator = Summarizer(4).load_from_checkpoint(translator_path)
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def pairwise_cosine_similarity(self, x, y):
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
        return x @ y.T

    def criterion(self, auc_cls, subword_f_expanded, tau=0.5):
        cos_dist = 1 - F.cosine_similarity(auc_cls, subword_f_expanded).mean()
        pw_cos_sim_auc = self.pairwise_cosine_similarity(auc_cls, auc_cls) / tau
        pw_cos_sim_sub = self.pairwise_cosine_similarity(subword_f_expanded, subword_f_expanded) / tau
        loss = cos_dist + F.mse_loss(pw_cos_sim_auc, pw_cos_sim_sub)
        return loss

    def forward(self, est_targets, embeds, ratios):
        n_src = est_targets.size(1)
        audio_embeds, word_embeds = [], []
        for i in range(n_src):
            auc_f = self.translator.get_acoustic_features(est_targets[:, i, :])
            auc_cls, subword_f_expanded = self.translator.run_and_match(auc_f, ratios[i], embeds[i])
            audio_embeds.append(auc_cls)
            word_embeds.append(subword_f_expanded)

        audio_embeds = torch.cat(audio_embeds, dim=0)
        word_embeds = torch.cat(word_embeds, dim=0)
        loss = self.criterion(audio_embeds, word_embeds)
        return loss


if __name__ == "__main__":
    import os
    import sys
    sys.path.append('../')
    from librimix_dataset import LibriMix
    from torch.utils.data import DataLoader
    from functools import partial
    from tqdm import tqdm
    from torch.nn.utils.rnn import pad_sequence
    
    summarizer_transformer = Summarizer(2).load_from_checkpoint(
        '<path/to/pretrained/weights>'  # Replace with the actual path to your pretrained weights
    )
    summarizer_transformer.eval()
    valid_set = LibriMix(
        csv_dir='../data/wav8k/min/dev', 
        task='sep_noisy', 
        sample_rate=8000, n_src=2, segment=3, 
        return_id=False, save_cls=False
    )
    valid_loader = DataLoader(
        valid_set, shuffle=False, batch_size=24, drop_last=True, collate_fn=partial(rnn_collate, n_src=2), num_workers=32, pin_memory=True)
    
    d_11, d_12, d_21, d_22 = [], [], [], []
    for batch in tqdm(valid_loader):
        m, s, [f1, f2], [r1, r2] = batch
        with torch.no_grad():
            auc_f1 = summarizer_transformer.get_acoustic_features(s[:, 0, :])
            auc_f2 = summarizer_transformer.get_acoustic_features(s[:, 1, :])
            
            auc_cls1, subword_f1_expanded = summarizer_transformer.run_and_match(auc_f1, r1, f1)
            auc_cls2, subword_f2_expanded = summarizer_transformer.run_and_match(auc_f2, r2, f2)
            min_len = min(auc_cls1.size(0), auc_cls2.size(0))

            d_11.append((auc_cls1 - subword_f1_expanded).pow(2).mean(1))
            d_12.append((auc_cls1[:min_len] - subword_f2_expanded[:min_len]).pow(2).mean(1))
            d_21.append((auc_cls2[:min_len] - subword_f1_expanded[:min_len]).pow(2).mean(1))
            d_22.append((auc_cls2 - subword_f2_expanded).pow(2).mean(1))

    d_11 = torch.cat(d_11, dim=0)
    d_12 = torch.cat(d_12, dim=0)
    d_21 = torch.cat(d_21, dim=0)
    d_22 = torch.cat(d_22, dim=0)