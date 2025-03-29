import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile

from wham_dataset import wham_noise_license
from transformers import BertTokenizer, BertModel

from tqdm import tqdm

MINI_URL = "https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1"


class LibriMix(Dataset):
    """Dataset class for LibriMix source separation tasks.

    Args:
        csv_dir (str): The path to the metadata file.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'`` :

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.

    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
    """

    dataset_name = "LibriMix"

    def __init__(
        self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, segment=3, return_id=False, save_cls=False
    ):
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        self.save_cls = save_cls
        # Get the csv corresponding to the task
        if task == "enh_single":
            md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "enh_both":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
            md_clean_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))
        elif task == "sep_clean":
            md_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "sep_noisy":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        self.n_src = n_src

        # add LM
        root_dirs = [
            '<path/to/LibriSpeech/dev-clean>',
            '<path/to/LibriSpeech/test-clean>',
            '<path/to/LibriSpeech/train-clean-100>',
            '<path/to/LibriSpeech/train-clean-360>',
        ]
        lm_type = 'bert-base-uncased'
        if self.save_cls:
            self.tokenizer = BertTokenizer.from_pretrained(lm_type)
            self.lm = BertModel.from_pretrained(lm_type)
            self.lm.eval()
            self.lm.cuda()
        # self.id2sentence, self.id2cls = self.build_transcript_dict(root_dirs)
        self.id2sentence, self.id2cls = self.build_alignment_dict(root_dirs)

        # drop mixture/sources pairs that are not included in the alignment files and reindex
        num_data = len(self.df)
        drop_indices = []
        for row, indices in enumerate(self.df['mixture_ID']):
            indices = indices.split('_')
            for idx in indices:
                if idx not in self.id2cls.keys():
                    drop_indices.append(row)
        num_drop = len(drop_indices)
        print(f'####### Drop {num_drop} pairs ({round(num_drop / num_data, 4) * 100}%)')
        self.df = self.df.drop(drop_indices)
        self.df.index = range(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        if 'test' in self.csv_dir:
            stop = row["length"]
        # If task is enh_both then the source is the clean mixture
        if "enh_both" in self.task:
            mix_clean_path = self.df_clean.iloc[idx]["mixture_path"]
            s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)

        else:
            # Read sources
            for i in range(self.n_src):
                source_path = row[f"source_{i + 1}_path"]
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
                sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        # 5400-34479-0005_4973-24515-0007.wav
        if self.n_src == 2:
            id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
            feats1 = self.id2cls[id1]['last_hidden_states']
            feats1, ratio1 = self.truncate_indices_and_return_duration_ratio(
                feats1,
                self.id2cls[id1]['end_indices'],
                start, stop
            )
            feats2 = self.id2cls[id2]['last_hidden_states']
            feats2, ratio2 = self.truncate_indices_and_return_duration_ratio(
                feats2,
                self.id2cls[id2]['end_indices'],
                start, stop
            )
        elif self.n_src == 3:
            id1, id2, id3 = mixture_path.split("/")[-1].split(".")[0].split("_")
            feats1 = self.id2cls[id1]['last_hidden_states']
            feats1, ratio1 = self.truncate_indices_and_return_duration_ratio(
                feats1,
                self.id2cls[id1]['end_indices'],
                start, stop
            )
            feats2 = self.id2cls[id2]['last_hidden_states']
            feats2, ratio2 = self.truncate_indices_and_return_duration_ratio(
                feats2,
                self.id2cls[id2]['end_indices'],
                start, stop
            )
            feats3 = self.id2cls[id3]['last_hidden_states']
            feats3, ratio3 = self.truncate_indices_and_return_duration_ratio(
                feats3,
                self.id2cls[id3]['end_indices'],
                start, stop
            )


        if not self.return_id:
            if self.n_src == 2:
                return mixture, sources, feats1, feats2, ratio1, ratio2
            elif self.n_src == 3:
                return mixture, sources, feats1, feats2, feats3, ratio1, ratio2, ratio3

        else:
            if self.n_src == 2:
                return mixture, sources, feats1, feats2, [id1, id2]
            elif self.n_src == 3:
                return mixture, sources, feats1, feats2, feats3, [id1, id2, id3]

    def truncate_indices_and_return_duration_ratio(self, feat, end_indices, start, stop):
        # from copy import deepcopy
        # tmp = deepcopy(end_indices)
        del_start = 0
        for i, eid in enumerate(end_indices):
            if start >= eid:
                del_start = i + 1
            if stop <= eid:
                del_stop = i
                break

        end_indices = end_indices[del_start:del_stop]
        feat = feat[del_start:del_stop + 1]
        if start not in end_indices:
            end_indices = [start] + end_indices
        if stop not in end_indices:
            end_indices = end_indices + [stop]
        end_indices = torch.tensor(end_indices)
        duration_indices = torch.diff(end_indices)
        # if not (0.5 * (duration_indices.sign() + 1)).all():
            # import pdb;pdb.set_trace()
        duration_ratio = duration_indices / duration_indices.sum()
        # if duration_ratio.shape[0] != len(feat):
            # print(duration_ratio.shape[0], feat.shape[0])
            # import pdb; pdb.set_trace()
        # assert duration_ratio.shape[0] != 0
        return feat, duration_ratio

    def build_alignment_dict(self, root_dirs):
        alignment_files = []
        id2sentence = dict()
        id2rep = dict()
        # get all aligned transcripts
        for root_dir in root_dirs:
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.endswith('.alignment.txt'):
                        alignment_files.append(os.path.join(root, f))
        # build dictionary with collected alignments
        print("Building alignment dictionary ...")
        for fn in tqdm(alignment_files):
            with open(fn, 'r') as f:
                lines = f.readlines()

            for line in lines:
                idx, words, end_times = line.strip().split(' ')
                words = words.replace('\"', '').split(',')
                words = ['[SEP]' if w == '' else w for w in words][1:-1] # drop start/end [SEP]
                sentence = ' '.join(words).strip(' ')
                end_indices = [int(float(e) * 16000) for e in end_times.replace('\"', '').split(',')]
                repeating_times = torch.diff(torch.tensor(end_indices), prepend=torch.zeros(1))
                repeating_times_sub = []

                # get and repeat embeddings
                id2sentence.update({idx: sentence})
                words = ['[CLS]'] + words + ['[SEP]']
                if self.save_cls:
                    with torch.no_grad():
                        reps = []
                        tokens = self.tokenizer(sentence, return_tensors='pt')
                        for wid, word in enumerate(words):
                            token = self.tokenizer(word, return_tensors='pt')
                            num_subwords = token['input_ids'].size(1) - 2
                            # print(word, num_subwords)
                            word_duration = repeating_times[wid].item()
                            subword_duration = int(word_duration // num_subwords)
                            subword_residual = int(word_duration % num_subwords)
                            for ns in range(num_subwords):
                                if ns == num_subwords - 1:
                                    repeating_times_sub.append(subword_duration + subword_residual) # if not complete divided
                                else:
                                    repeating_times_sub.append(subword_duration) # complete division
                        end_indices = torch.cumsum(torch.tensor(repeating_times_sub), dim=0).tolist()
                        for k, v in tokens.items():
                            tokens[k] = v.cuda()   
                        reps = self.lm(**tokens).get('last_hidden_state')
                        reps = reps.detach().cpu()[0]
                    assert reps.shape[0] == len(end_indices)
                    id2rep.update(
                        {
                            idx: {'last_hidden_states': reps, 'end_indices': end_indices}
                        }
                    )
        if self.save_cls:
            torch.save(id2rep, 'id2rep.pth')
        else:
            id2rep = torch.load('id2rep.pth')
        return id2sentence, id2rep

    def build_transcript_dict(self, root_dirs):
        transcript_files = []
        id2sentence = dict()
        id2rep = dict()
        # get all transcripts
        for root_dir in root_dirs:
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.endswith('.txt'):
                        transcript_files.append(os.path.join(root, f))
        # build dictionary with collected transcripts
        print("Building sentence embedding dictionary ...")
        for fn in tqdm(transcript_files):
            with open(fn, 'r') as f:
                pairs = [line.rstrip().split(' ', 1) for line in f.read().splitlines()]
                for idx, sentence in pairs:
                    id2sentence.update({idx: sentence})
                    if self.save_cls:
                        # Get Text Fetures
                        sentence = id2sentence.get(idx)
                        with torch.no_grad():
                            tokens = self.tokenizer(sentence, return_tensors='pt')
                            for k, v in tokens.items():
                                tokens[k] = v.cuda()
                            rep = self.lm(**tokens).get('last_hidden_state')
                            rep = rep.detach().cpu()
                        id2rep.update({idx: rep})
        if self.save_cls:
            torch.save(id2rep, 'id2rep.pth')
        else:
            id2rep = torch.load('id2rep.pth')
        return id2sentence, id2rep

    def build_transcript_dict_cls(self, root_dirs):
        transcript_files = []
        id2sentence = dict()
        id2cls = dict()
        # get all transcripts
        for root_dir in root_dirs:
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.endswith('.txt'):
                        transcript_files.append(os.path.join(root, f))
        # build dictionary with collected transcripts
        print("Building sentence embedding dictionary ...")
        for fn in tqdm(transcript_files):
            with open(fn, 'r') as f:
                pairs = [line.rstrip().split(' ', 1) for line in f.read().splitlines()]
                for idx, sentence in pairs:
                    id2sentence.update({idx: sentence})
                    if self.save_cls:
                        # Get Text Fetures
                        sentence = id2sentence.get(idx)
                        with torch.no_grad():
                            tokens = self.tokenizer(sentence, return_tensors='pt')
                            for k, v in tokens.items():
                                tokens[k] = v.cuda()
                            cls_rep = self.lm(**tokens).get('last_hidden_state')[:, 0, :]
                            cls_rep = cls_rep.detach().cpu()
                        id2cls.update({idx: cls_rep})
        if self.save_cls:
            torch.save(id2cls, 'id2cls.pth')
        else:
            id2cls = torch.load('id2cls.pth')
        return id2sentence, id2cls

    @classmethod
    def loaders_from_mini(cls, batch_size=4, **kwargs):
        """Downloads MiniLibriMix and returns train and validation DataLoader.

        Args:
            batch_size (int): Batch size of the Dataloader. Only DataLoader param.
                To have more control on Dataloader, call `mini_from_download` and
                instantiate the DatalLoader.
            **kwargs: keyword arguments to pass the `LibriMix`, see `__init__`.
                The kwargs will be fed to both the training set and validation
                set.

        Returns:
            train_loader, val_loader: training and validation DataLoader out of
            `LibriMix` Dataset.

        Examples
            >>> from asteroid.data import LibriMix
            >>> train_loader, val_loader = LibriMix.loaders_from_mini(
            >>>     task='sep_clean', batch_size=4
            >>> )
        """
        train_set, val_set = cls.mini_from_download(**kwargs)
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        return train_loader, val_loader

    @classmethod
    def mini_from_download(cls, **kwargs):
        """Downloads MiniLibriMix and returns train and validation Dataset.
        If you want to instantiate the Dataset by yourself, call
        `mini_download` that returns the path to the path to the metadata files.

        Args:
            **kwargs: keyword arguments to pass the `LibriMix`, see `__init__`.
                The kwargs will be fed to both the training set and validation
                set

        Returns:
            train_set, val_set: training and validation instances of
            `LibriMix` (data.Dataset).

        Examples
            >>> from asteroid.data import LibriMix
            >>> train_set, val_set = LibriMix.mini_from_download(task='sep_clean')
        """
        # kwargs checks
        assert "csv_dir" not in kwargs, "Cannot specify csv_dir when downloading."
        assert kwargs.get("task", "sep_clean") in [
            "sep_clean",
            "sep_noisy",
        ], "Only clean and noisy separation are supported in MiniLibriMix."
        assert (
            kwargs.get("sample_rate", 8000) == 8000
        ), "Only 8kHz sample rate is supported in MiniLibriMix."
        # Download LibriMix in current directory
        meta_path = cls.mini_download()
        # Create dataset instances
        train_set = cls(os.path.join(meta_path, "train"), sample_rate=8000, **kwargs)
        val_set = cls(os.path.join(meta_path, "val"), sample_rate=8000, **kwargs)
        return train_set, val_set

    @staticmethod
    def mini_download():
        """Downloads MiniLibriMix from Zenodo in current directory

        Returns:
            The path to the metadata directory.
        """
        mini_dir = "./MiniLibriMix/"
        os.makedirs(mini_dir, exist_ok=True)
        # Download zip (or cached)
        zip_path = mini_dir + "MiniLibriMix.zip"
        if not os.path.isfile(zip_path):
            hub.download_url_to_file(MINI_URL, zip_path)
        # Unzip zip
        cond = all([os.path.isdir("MiniLibriMix/" + f) for f in ["train", "val", "metadata"]])
        if not cond:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("./")  # Will unzip in MiniLibriMix
        # Reorder metadata
        src = "MiniLibriMix/metadata/"
        for mode in ["train", "val"]:
            dst = f"MiniLibriMix/metadata/{mode}/"
            os.makedirs(dst, exist_ok=True)
            [
                shutil.copyfile(src + f, dst + f)
                for f in os.listdir(src)
                if mode in f and os.path.isfile(src + f)
            ]
        return "./MiniLibriMix/metadata"

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self._dataset_name()
        infos["task"] = self.task
        if self.task == "sep_clean":
            data_license = [librispeech_license]
        else:
            data_license = [librispeech_license, wham_noise_license]
        infos["licenses"] = data_license
        return infos

    def _dataset_name(self):
        """Differentiate between 2 and 3 sources."""
        return f"Libri{self.n_src}Mix"


librispeech_license = dict(
    title="LibriSpeech ASR corpus",
    title_link="http://www.openslr.org/12",
    author="Vassil Panayotov",
    author_link="https://github.com/vdp",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
