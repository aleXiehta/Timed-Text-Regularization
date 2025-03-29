import os
import sys
import math
sys.path.append('../')
from librimix_dataset import LibriMix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoProcessor, Wav2Vec2Model, HubertModel, WavLMModel
from torchaudio.functional import resample
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse


def rnn_collate(batch, n_src):
    # n_src = 2 if len(batch) == 4 else 3
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
        return m, s, [f1, f2], [r1, r2]


class SSLPreTrained(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ssl_pretrained = WavLMModel.from_pretrained("microsoft/wavlm-base")
        # self.ssl_pretrained = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    
    def forward(self, x):
        return self.ssl_pretrained(x)


class Summarizer(pl.LightningModule):
    def __init__(
        self, 
        num_layers=4, 
        d_model=768,
        coeff1 = 1.0,
        coeff2 = 1.0,
        freeze_ssl = False,
    ):
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
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        if freeze_ssl:
            self.ssl_pretrained.freeze()
        # self.ce_loss = nn.CrossEntropyLoss()
        # self.pe = PositionalEncoding(d_model)
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        # self.layer_norm = nn.LayerNorm([149, 768])
        
    def get_acoustic_features(self, audio):
        with torch.no_grad():
            B = audio.size(0)
            out = self.ssl_pretrained(resample(audio, orig_freq=8000, new_freq=16000).view(B, -1))
        # return self.layer_norm(out.last_hidden_state)
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

    def get_grid_mask(self, len1, len2):
        num_subwords_1 = torch.sum(len1 != 0, dim=1)
        num_subwords_2 = torch.sum(len2 != 0, dim=1)
        grid_boundaries = torch.cat(
            [
                torch.zeros(1, device=len1.device).int(),
                num_subwords_1,
                num_subwords_2,
            ],
            dim=0
        ).cumsum(0)
        size = int(grid_boundaries[-1].item())
        mask = torch.zeros(size, size, device=len1.device)
        for i in range(len(grid_boundaries) - 1):
            st, ed = grid_boundaries[i], grid_boundaries[i + 1]
            mask[st:ed, st:ed] = 1.0
        return mask

    def run_and_match(self, auc_f, repeat_ratios, subword_f, get_sum_out=False):
        sub_auc_f, subword_lengths = self.sentence2subwords(auc_f, repeat_ratios)
        # sub_auc_f = []
        # for sf, sf_len in zip(subword_f, subword_lengths):
        #     sub_auc_f.append(sf.repeat_interleave(sf_len.int(), dim=0))
        # sub_auc_f = torch.stack(sub_auc_f, dim=0)
        sub_auc_f = torch.cat([self.cls.repeat(sub_auc_f.shape[0], 1, 1), sub_auc_f], dim=1)
        
        # src_mask = torch.zeros(sub_auc_f.size(0), sub_auc_f.size(1)).to(sub_auc_f.device)
        # src_mask[sub_auc_f.sum(-1) == 0] = -torch.inf
        auc_sum = self.translator(sub_auc_f)[:, 0, :]
        # auc_cls = self.translator(self.pos_embed(sub_auc_f))[:, 0, :]
        subword_f_expanded = []
        start = 0
        for i, (swf, r) in enumerate(zip(subword_f.view(-1, self.d_model), subword_lengths.view(-1))):
            r = int(r.item())
            if r > 0:
                subword_f_expanded.append(swf)
        subword_f_expanded = torch.stack(subword_f_expanded, dim=0)
        # subword_lengths = subword_lengths.view(-1)
        # subword_lengths = subword_lengths[subword_lengths.nonzero()]
        # print(subword_lengths.shape, sub_auc_f.shape, subword_lengths.shape)
        # auc_summary = sub_auc_f.sum(1) / subword_lengths
        auc_agg = self.inter(auc_sum.unsqueeze(0)).squeeze(0)
        if not get_sum_out:
            return auc_agg, subword_f_expanded, subword_lengths
        else:
            return auc_sum, auc_agg, subword_f_expanded, subword_lengths
    
    def pairwise_cosine_similarity(self, x, y):
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
        return x @ y.T

    def masked_mse_loss(self, y_hat, y, mask):
        masked_squared_error = torch.square(y_hat - y) * mask
        masked_mse = torch.sum(masked_squared_error) / mask.sum()
        return masked_mse
        
    def criterion(self, auc_cls, subword_f_expanded, mask):
        cos_dist = 1 - F.cosine_similarity(auc_cls, subword_f_expanded).mean()
        pw_cos_sim_auc = self.pairwise_cosine_similarity(auc_cls, auc_cls)
        pw_cos_sim_sub = self.pairwise_cosine_similarity(subword_f_expanded, subword_f_expanded)
        
        # logits_auc = torch.nn.functional.log_softmax(pw_cos_sim_auc, dim=1)
        # logits_sub = torch.nn.functional.log_softmax(pw_cos_sim_sub, dim=1)
        loss = self.coeff1 * cos_dist + self.coeff2 * self.masked_mse_loss(pw_cos_sim_auc, pw_cos_sim_sub, mask)
        # loss = -logits.mean()
        # loss = self.ce_loss(logits, torch.arange(y_hat.size(0)).to(self.device))
        return loss

    def training_step(self, batch, batch_idx):
        m, s, [f1, f2], [r1, r2] = batch
        auc_f1 = self.get_acoustic_features(s[:, 0, :])
        auc_f2 = self.get_acoustic_features(s[:, 1, :])
        auc_cls1, subword_f1_expanded, len1 = self.run_and_match(auc_f1, r1, f1)
        auc_cls2, subword_f2_expanded, len2 = self.run_and_match(auc_f2, r2, f2)
        mask = self.get_grid_mask(len1, len2)
        loss = self.criterion(
            torch.cat([auc_cls1, auc_cls2], dim=0), 
            torch.cat([subword_f1_expanded, subword_f2_expanded], dim=0),
            mask=mask
        )
        # loss = self.criterion(torch.cat([auc_cls1, auc_cls2], dim=0), torch.cat([subword_f1_expanded, subword_f2_expanded], dim=0))
        # loss = 0.5 * self.criterion(auc_cls1, subword_f1_expanded) + 0.5 * self.criterion(auc_cls2, subword_f2_expanded)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        m, s, [f1, f2], [r1, r2] = batch
        auc_f1 = self.get_acoustic_features(s[:, 0, :])
        auc_f2 = self.get_acoustic_features(s[:, 1, :])
        auc_cls1, subword_f1_expanded, len1 = self.run_and_match(auc_f1, r1, f1)
        auc_cls2, subword_f2_expanded, len2 = self.run_and_match(auc_f2, r2, f2)
        mask = self.get_grid_mask(len1, len2)
        loss = self.criterion(
            torch.cat([auc_cls1, auc_cls2], dim=0), 
            torch.cat([subword_f1_expanded, subword_f2_expanded], dim=0),
            mask=mask
        )
        # print(auc_cls1.shape, subword_f1_expanded.shape)
        # loss = self.criterion(torch.cat([auc_cls1, auc_cls2], dim=0), torch.cat([subword_f1_expanded, subword_f2_expanded], dim=0))
        # loss = 0.5 * self.criterion(auc_cls1, subword_f1_expanded) + 0.5 * self.criterion(auc_cls2, subword_f2_expanded)
        self.log("valid_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-4, betas=(0.9, 0.98))
        return optimizer
    
    def forward(self, batch):
        m, s, [f1, f2], [r1, r2] = batch
        auc_f1 = self.get_acoustic_features(s[:, 0, :])
        auc_f2 = self.get_acoustic_features(s[:, 1, :])
        auc_cls1, subword_f1_expanded, len1 = self.run_and_match(auc_f1, r1, f1)
        auc_cls2, subword_f2_expanded, len2 = self.run_and_match(auc_f2, r2, f2)
        mask = self.get_grid_mask(len1, len2)
        loss = self.criterion(
            torch.cat([auc_cls1, auc_cls2], dim=0), 
            torch.cat([subword_f1_expanded, subword_f2_expanded], dim=0),
            mask=mask
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--num_layers", type=int, default=2)
    parser.add_argument("-a", "--alpha", type=float, default=1.0)
    parser.add_argument("-b", "--beta", type=float, default=1.0)
    parser.add_argument("-e", "--exp_name", type=str, default="")
    parser.add_argument("-f", "--freeze_ssl", action="store_true")
    args = parser.parse_args()

    train_set = LibriMix(
        csv_dir='<path/to/wav8k/min/train-360>', 
        task='sep_clean', 
        sample_rate=8000, n_src=3, segment=3, 
        return_id=False, save_cls=False
    )
    valid_set = LibriMix(
        csv_dir='<path/to/data/wav8k/min/dev>', 
        task='sep_clean', 
        sample_rate=8000, n_src=3, segment=3, 
        return_id=False, save_cls=False
    )
    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=12, drop_last=True, collate_fn=partial(rnn_collate, n_src=3), num_workers=32, pin_memory=True)
    valid_loader = DataLoader(
        valid_set, shuffle=False, batch_size=12, drop_last=True, collate_fn=partial(rnn_collate, n_src=3), num_workers=32, pin_memory=True)
    
    summarizer_transformer = Summarizer(
        num_layers=args.num_layers, 
        coeff1=args.alpha, 
        coeff2=args.beta,
        freeze_ssl=args.freeze_ssl,
    )

    exp_name = args.exp_name
    logger = pl.loggers.TensorBoardLogger('tb_logs', name=exp_name)

    exp_dir = 'exp/'
    checkpoint_dir = os.path.join(exp_dir, exp_name, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, 
        monitor="valid_loss",
        mode="min", save_top_k=1, verbose=True
    )

    callbacks = []
    callbacks.append(checkpoint)
    callbacks.append(
        EarlyStopping(
            monitor="valid_loss", mode="min", 
            patience=30, verbose=True
        )
    )

    trainer=pl.Trainer(
        max_epochs=100,
        logger=logger,
        strategy='dp',
        devices='auto',
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu"
    )
    trainer.fit(summarizer_transformer, train_dataloaders=train_loader, val_dataloaders=valid_loader)