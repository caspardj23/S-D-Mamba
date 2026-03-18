import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val", "recursive"]
        type_map = {"train": 0, "val": 1, "test": 2, "recursive": 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        if self.target in cols:
            cols.remove(self.target)
        cols.remove("date")
        if self.target in df_raw.columns:
            df_raw = df_raw[["date"] + cols + [self.target]]
        else:
            df_raw = df_raw[["date"] + cols]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
            0,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data["data"][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[: int(train_ratio * len(data))]
        valid_data = data[
            int(train_ratio * len(data)) : int((train_ratio + valid_ratio) * len(data))
        ]
        test_data = data[int((train_ratio + valid_ratio) * len(data)) :]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = (
            df.fillna(method="ffill", limit=len(df))
            .fillna(method="bfill", limit=len(df))
            .values
        )

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(
            os.path.join(self.root_path, self.data_path), "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                line = line.strip("\n").split(",")
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
            0,
        ]
        border2s = [num_train, num_train + num_valid, len(df_raw), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(
        self,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="15min",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["pred"]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        if self.cols:
            cols = self.cols.copy()
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            if self.target in cols:
                cols.remove(self.target)
            cols.remove("date")
        if self.target in df_raw.columns:
            df_raw = df_raw[["date"] + cols + [self.target]]
        else:
            df_raw = df_raw[["date"] + cols]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq
        )

        df_stamp = pd.DataFrame(columns=["date"])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin : r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin : r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Haskins_MAE(Dataset):
    """
    Haskins EMA dataset with sentence-aware splitting and windowing.

    Expects a CSV with columns: sentence_id, speaker_id, time_s, 0, 1, ..., N
    Each sentence is a separate utterance identified by sentence_id.

    Key design decisions:
      1. Splitting at the sentence level: for each speaker, 70% of sentences go
         to train, 10% to val, 20% to test. No sentence is split across sets.
      2. Windows stay within sentences: no window ever crosses a sentence boundary,
         so the model never sees discontinuous articulation.
      3. Window size = seq_len only: both MAE pretrain (which ignores batch_y) and
         finetune (which carves the target from batch_x) only need seq_len frames.

    Args:
        root_path: Directory containing the CSV
        flag: 'train', 'val', or 'test'
        size: [seq_len, label_len, pred_len]
        features: 'M' for multivariate (default)
        data_path: CSV filename
        target: target column name (unused for MAE, kept for interface compat)
        scale: whether to apply StandardScaler (fit on train split)
        timeenc: time encoding type
        freq: frequency string
        stride: window stride in frames (default 80)
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="ema_8_pos_xz.csv",
        target="15",
        scale=True,
        timeenc=1,
        freq="h",
        stride=80,
    ):
        if size is None:
            self.seq_len = 160
            self.label_len = 0
            self.pred_len = 160
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.stride = stride

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Extract feature columns (everything except metadata)
        meta_cols = {"sentence_id", "sentence_group", "speaker_id", "time_s", "phone", "date"}
        if self.features == "M" or self.features == "MS":
            feature_cols = [c for c in df_raw.columns if c not in meta_cols]
        elif self.features == "S":
            feature_cols = [self.target]

        all_data = df_raw[feature_cols].values.astype(np.float32)
        sentence_ids = df_raw["sentence_id"].values
        speaker_ids = df_raw["speaker_id"].values

        # Store phone labels if available (for visualization)
        if "phone" in df_raw.columns:
            self.phone_labels = df_raw["phone"].values
        else:
            self.phone_labels = None

        # Store per-frame sentence_id for metadata lookup
        self.sentence_ids_per_frame = sentence_ids.copy()

        # Load sentence text lookup from ema_7_sentences.csv if it exists
        self.sentence_text_map = {}
        sentences_csv = os.path.join(self.root_path, "ema_7_sentences.csv")
        if os.path.exists(sentences_csv):
            df_sent = pd.read_csv(sentences_csv)
            for _, row in df_sent.iterrows():
                self.sentence_text_map[int(row["sentence_id"])] = {
                    "sentence": row["sentence"],
                    "speaker_id": row["speaker_id"],
                    "sentence_group": row["sentence_group"],
                }

        # Build sentence table: sentence_id → (start_row, end_row, speaker, group)
        sentences = {}
        if "sentence_group" in df_raw.columns:
            sentence_groups = df_raw["sentence_group"].values
            for sid in np.unique(sentence_ids):
                rows = np.where(sentence_ids == sid)[0]
                sentences[sid] = (rows[0], rows[-1] + 1, speaker_ids[rows[0]],
                                  sentence_groups[rows[0]])
        else:
            for sid in np.unique(sentence_ids):
                rows = np.where(sentence_ids == sid)[0]
                sentences[sid] = (rows[0], rows[-1] + 1, speaker_ids[rows[0]], sid)

        # Split by sentence_group to prevent data leakage across speakers
        # All sentences with the same group (same text, different speakers/reps)
        # go into the same split
        all_groups = sorted(set(grp for _, _, _, grp in sentences.values()))
        n_groups = len(all_groups)
        n_train_g = int(n_groups * 0.7)
        n_test_g = int(n_groups * 0.2)
        n_val_g = n_groups - n_train_g - n_test_g

        train_groups = set(all_groups[:n_train_g])
        val_groups = set(all_groups[n_train_g:n_train_g + n_val_g])
        test_groups = set(all_groups[n_train_g + n_val_g:])

        train_sents = []
        val_sents = []
        test_sents = []
        n_speakers = len(set(speaker_ids))

        for sid, (start, end, speaker, grp) in sentences.items():
            entry = (sid, start, end)
            if grp in train_groups:
                train_sents.append(entry)
            elif grp in val_groups:
                val_sents.append(entry)
            else:
                test_sents.append(entry)

        # Sort for reproducibility
        train_sents.sort(key=lambda x: x[0])
        val_sents.sort(key=lambda x: x[0])
        test_sents.sort(key=lambda x: x[0])

        # Fit scaler on training sentences only
        if self.scale:
            train_chunks = [all_data[s:e] for _, s, e in train_sents]
            train_all = np.concatenate(train_chunks, axis=0)
            self.scaler.fit(train_all)
            all_data = self.scaler.transform(all_data)

        # Select sentences for this split
        if self.flag == "train":
            split_sents = train_sents
        elif self.flag == "val":
            split_sents = val_sents
        else:
            split_sents = test_sents

        # Build window start positions within sentences
        window_starts = []
        window_len = self.seq_len  # only need seq_len frames per window
        skipped = 0

        for sid, seg_start, seg_end in split_sents:
            seg_len = seg_end - seg_start
            if seg_len < window_len:
                skipped += 1
                continue
            n_windows = (seg_len - window_len) // self.stride + 1
            for w in range(n_windows):
                window_starts.append(seg_start + w * self.stride)

        self.window_starts = np.array(window_starts, dtype=np.int64)
        self.all_data = all_data

        # Dummy timestamps (interface required, not used by MAE)
        self.data_stamp = np.zeros((len(all_data), 4), dtype=np.float32)

        print(f"  [Dataset_Haskins_MAE] {n_speakers} speakers, "
              f"{len(sentences)} total sentences, "
              f"{n_groups} sentence groups, "
              f"{len(all_data)} total frames, "
              f"{all_data.shape[1]} variates")
        print(f"  [Dataset_Haskins_MAE] split by sentence_group: "
              f"train={len(train_groups)}g/{len(train_sents)}s, "
              f"val={len(val_groups)}g/{len(val_sents)}s, "
              f"test={len(test_groups)}g/{len(test_sents)}s")
        print(f"  [Dataset_Haskins_MAE] {self.flag}: {len(split_sents)} sentences, "
              f"{len(self.window_starts)} windows "
              f"(stride={self.stride}, seq_len={self.seq_len}, skipped={skipped})")

    def __getitem__(self, index):
        start = self.window_starts[index]
        s_end = start + self.seq_len

        seq_x = self.all_data[start:s_end]
        seq_y = seq_x  # target carved from seq_x in experiment code
        seq_x_mark = self.data_stamp[start:s_end]
        seq_y_mark = seq_x_mark

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.window_starts)

    def get_phone_labels(self, index):
        """Return phone labels for a given window index, or None if unavailable."""
        if self.phone_labels is None:
            return None
        start = self.window_starts[index]
        s_end = start + self.seq_len
        return self.phone_labels[start:s_end]

    def get_sentence_info(self, index):
        """Return sentence text and metadata for a given window index, or None."""
        start = self.window_starts[index]
        sid = int(self.sentence_ids_per_frame[start])
        if sid in self.sentence_text_map:
            info = self.sentence_text_map[sid]
            return {
                "sentence_id": sid,
                "sentence": info["sentence"],
                "speaker_id": info["speaker_id"],
            }
        return None

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Haskins_Probe(Dataset):
    """
    Haskins EMA dataset for probing encoder representations.

    Returns per-frame windows of EMA data with classification labels.
    Supports multiple probe tasks:
      - 'speaker': classify speaker identity (8 classes, labels from CSV)
      - 'phoneme': classify phoneme identity (~40 classes, from external label file)
      - 'manner': classify manner of articulation (~7 classes, derived from phoneme)

    Phoneme/manner probes require a pre-computed label file (NumPy .npz) with
    per-frame phoneme IDs aligned to the EMA CSV rows. See
    scripts/prepare_haskins_phoneme_labels.py to generate this file.

    Uses the same sentence-aware per-speaker 70/10/20 split as Dataset_Haskins_MAE.
    """

    # ARPABET manner-of-articulation mapping
    MANNER_MAP = {
        # Stops
        'B': 'stop', 'D': 'stop', 'G': 'stop', 'K': 'stop', 'P': 'stop', 'T': 'stop',
        # Fricatives
        'CH': 'fricative', 'DH': 'fricative', 'F': 'fricative', 'HH': 'fricative',
        'JH': 'fricative', 'S': 'fricative', 'SH': 'fricative', 'TH': 'fricative',
        'V': 'fricative', 'Z': 'fricative', 'ZH': 'fricative',
        # Nasals
        'M': 'nasal', 'N': 'nasal', 'NG': 'nasal',
        # Approximants
        'L': 'approximant', 'R': 'approximant', 'W': 'approximant', 'Y': 'approximant',
        # Vowels (all ARPABET vowels including diphthongs)
        'AA': 'vowel', 'AE': 'vowel', 'AH': 'vowel', 'AO': 'vowel', 'AW': 'vowel',
        'AY': 'vowel', 'EH': 'vowel', 'ER': 'vowel', 'EY': 'vowel', 'IH': 'vowel',
        'IY': 'vowel', 'OW': 'vowel', 'OY': 'vowel', 'UH': 'vowel', 'UW': 'vowel',
        # Silence
        'SIL': 'silence', 'sp': 'silence', 'sil': 'silence', '': 'silence',
    }
    MANNER_CLASSES = ['silence', 'stop', 'fricative', 'nasal', 'approximant', 'vowel']

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="ema_8_pos_xz.csv",
        target="15",
        scale=True,
        timeenc=1,
        freq="h",
        stride=80,
        probe_task="speaker",
        phoneme_label_path=None,
    ):
        if size is None:
            self.seq_len = 160
            self.label_len = 0
            self.pred_len = 160
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "val", "test"]
        assert probe_task in ["speaker", "phoneme", "manner"]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.stride = stride
        self.probe_task = probe_task
        self.phoneme_label_path = phoneme_label_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Extract feature columns
        meta_cols = {"sentence_id", "sentence_group", "speaker_id", "time_s", "phone", "date"}
        if self.features == "M" or self.features == "MS":
            feature_cols = [c for c in df_raw.columns if c not in meta_cols]
        elif self.features == "S":
            feature_cols = [self.target]

        all_data = df_raw[feature_cols].values.astype(np.float32)
        sentence_ids = df_raw["sentence_id"].values
        speaker_ids = df_raw["speaker_id"].values

        # Build speaker label mapping
        unique_speakers = sorted(set(speaker_ids))
        self.speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
        self.num_speakers = len(unique_speakers)
        speaker_labels = np.array([self.speaker_to_idx[s] for s in speaker_ids], dtype=np.int64)

        # Load phoneme labels if needed
        if self.probe_task in ("phoneme", "manner"):
            if self.phoneme_label_path is None:
                # Try default path
                self.phoneme_label_path = os.path.join(
                    self.root_path, "phoneme_labels.npz"
                )
            if not os.path.exists(self.phoneme_label_path):
                raise FileNotFoundError(
                    f"Phoneme label file not found: {self.phoneme_label_path}\n"
                    f"Run scripts/prepare_haskins_phoneme_labels.py first to generate it."
                )
            label_data = np.load(self.phoneme_label_path, allow_pickle=True)
            phoneme_ids = label_data["phoneme_ids"]  # [N_frames] int array
            self.phoneme_names = label_data["phoneme_names"].tolist()  # list of phone strings
            self.num_phonemes = len(self.phoneme_names)

            if self.probe_task == "manner":
                # Map phoneme names to manner classes
                manner_to_idx = {m: i for i, m in enumerate(self.MANNER_CLASSES)}
                phoneme_to_manner = np.zeros(self.num_phonemes, dtype=np.int64)
                for pid, pname in enumerate(self.phoneme_names):
                    # Strip stress digits (e.g., 'AH0' -> 'AH')
                    clean = pname.rstrip('0123456789')
                    manner = self.MANNER_MAP.get(clean, 'silence')
                    phoneme_to_manner[pid] = manner_to_idx[manner]
                self.frame_labels = phoneme_to_manner[phoneme_ids]
                self.num_classes = len(self.MANNER_CLASSES)
                self.class_names = self.MANNER_CLASSES
            else:
                self.frame_labels = phoneme_ids
                self.num_classes = self.num_phonemes
                self.class_names = self.phoneme_names
        else:
            # Speaker task: label is per-window (constant within sentence)
            self.frame_labels = speaker_labels
            self.num_classes = self.num_speakers
            self.class_names = unique_speakers

        # Build sentence table: sentence_id → (start_row, end_row, speaker, group)
        sentences = {}
        if "sentence_group" in df_raw.columns:
            sentence_groups = df_raw["sentence_group"].values
            for sid in np.unique(sentence_ids):
                rows = np.where(sentence_ids == sid)[0]
                sentences[sid] = (rows[0], rows[-1] + 1, speaker_ids[rows[0]],
                                  sentence_groups[rows[0]])
        else:
            for sid in np.unique(sentence_ids):
                rows = np.where(sentence_ids == sid)[0]
                sentences[sid] = (rows[0], rows[-1] + 1, speaker_ids[rows[0]], sid)

        # Split by sentence_group to prevent data leakage across speakers
        all_groups = sorted(set(grp for _, _, _, grp in sentences.values()))
        n_groups = len(all_groups)
        n_train_g = int(n_groups * 0.7)
        n_test_g = int(n_groups * 0.2)
        n_val_g = n_groups - n_train_g - n_test_g

        train_groups = set(all_groups[:n_train_g])
        val_groups = set(all_groups[n_train_g:n_train_g + n_val_g])
        test_groups = set(all_groups[n_train_g + n_val_g:])

        train_sents, val_sents, test_sents = [], [], []
        n_speakers = len(set(speaker_ids))
        for sid, (start, end, speaker, grp) in sentences.items():
            entry = (sid, start, end)
            if grp in train_groups:
                train_sents.append(entry)
            elif grp in val_groups:
                val_sents.append(entry)
            else:
                test_sents.append(entry)

        train_sents.sort(key=lambda x: x[0])
        val_sents.sort(key=lambda x: x[0])
        test_sents.sort(key=lambda x: x[0])

        # Fit scaler on training data
        if self.scale:
            train_chunks = [all_data[s:e] for _, s, e in train_sents]
            train_all = np.concatenate(train_chunks, axis=0)
            self.scaler.fit(train_all)
            all_data = self.scaler.transform(all_data)

        if self.flag == "train":
            split_sents = train_sents
        elif self.flag == "val":
            split_sents = val_sents
        else:
            split_sents = test_sents

        # Build windows
        window_starts = []
        window_len = self.seq_len
        skipped = 0
        for sid, seg_start, seg_end in split_sents:
            seg_len = seg_end - seg_start
            if seg_len < window_len:
                skipped += 1
                continue
            n_windows = (seg_len - window_len) // self.stride + 1
            for w in range(n_windows):
                window_starts.append(seg_start + w * self.stride)

        self.window_starts = np.array(window_starts, dtype=np.int64)
        self.all_data = all_data
        self.all_labels = self.frame_labels

        print(f"  [Dataset_Haskins_Probe] task={self.probe_task}, "
              f"num_classes={self.num_classes}, "
              f"speakers={self.num_speakers}")
        print(f"  [Dataset_Haskins_Probe] {self.flag}: "
              f"{len(split_sents)} sentences, "
              f"{len(self.window_starts)} windows "
              f"(stride={self.stride}, seq_len={self.seq_len}, skipped={skipped})")

    def __getitem__(self, index):
        start = self.window_starts[index]
        s_end = start + self.seq_len

        seq_x = self.all_data[start:s_end]
        labels = self.all_labels[start:s_end]

        return seq_x, labels

    def __len__(self):
        return len(self.window_starts)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Haskins_Forecast(Dataset):
    """
    Haskins EMA dataset for multivariate forecasting with sentence-aware splitting.

    Expects a CSV with columns: sentence_id, speaker_id, time_s, 0, 1, ..., N
    Windows never cross sentence boundaries (no discontinuous articulation).

    Splitting: per speaker, 70% train / 10% val / 20% test at sentence level.
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="ema_8_pos_xz.csv",
        target="15",
        scale=True,
        timeenc=1,
        freq="h",
        stride=1,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.stride = stride

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        meta_cols = {"sentence_id", "sentence_group", "speaker_id", "time_s", "phone", "date"}
        if self.features == "M" or self.features == "MS":
            feature_cols = [c for c in df_raw.columns if c not in meta_cols]
        elif self.features == "S":
            feature_cols = [self.target]

        all_data = df_raw[feature_cols].values.astype(np.float32)
        sentence_ids = df_raw["sentence_id"].values
        speaker_ids = df_raw["speaker_id"].values

        # Build sentence table: sentence_id → (start_row, end_row, speaker, group)
        sentences = {}
        if "sentence_group" in df_raw.columns:
            sentence_groups = df_raw["sentence_group"].values
            for sid in np.unique(sentence_ids):
                rows = np.where(sentence_ids == sid)[0]
                sentences[sid] = (rows[0], rows[-1] + 1, speaker_ids[rows[0]],
                                  sentence_groups[rows[0]])
        else:
            for sid in np.unique(sentence_ids):
                rows = np.where(sentence_ids == sid)[0]
                sentences[sid] = (rows[0], rows[-1] + 1, speaker_ids[rows[0]], sid)

        # Split by sentence_group to prevent data leakage across speakers
        all_groups = sorted(set(grp for _, _, _, grp in sentences.values()))
        n_groups = len(all_groups)
        n_train_g = int(n_groups * 0.7)
        n_test_g = int(n_groups * 0.2)
        n_val_g = n_groups - n_train_g - n_test_g

        train_groups = set(all_groups[:n_train_g])
        val_groups = set(all_groups[n_train_g:n_train_g + n_val_g])
        test_groups = set(all_groups[n_train_g + n_val_g:])

        train_sents, val_sents, test_sents = [], [], []
        n_speakers = len(set(speaker_ids))
        for sid, (start, end, speaker, grp) in sentences.items():
            entry = (sid, start, end)
            if grp in train_groups:
                train_sents.append(entry)
            elif grp in val_groups:
                val_sents.append(entry)
            else:
                test_sents.append(entry)

        train_sents.sort(key=lambda x: x[0])
        val_sents.sort(key=lambda x: x[0])
        test_sents.sort(key=lambda x: x[0])

        # Fit scaler on training sentences only
        if self.scale:
            train_chunks = [all_data[s:e] for _, s, e in train_sents]
            train_all = np.concatenate(train_chunks, axis=0)
            self.scaler.fit(train_all)
            all_data = self.scaler.transform(all_data)

        if self.flag == "train":
            split_sents = train_sents
        elif self.flag == "val":
            split_sents = val_sents
        else:
            split_sents = test_sents

        # Build window start positions; each window spans seq_len + pred_len frames
        window_len = self.seq_len + self.pred_len
        window_starts = []
        skipped = 0
        for sid, seg_start, seg_end in split_sents:
            seg_len = seg_end - seg_start
            if seg_len < window_len:
                skipped += 1
                continue
            n_windows = (seg_len - window_len) // self.stride + 1
            for w in range(n_windows):
                window_starts.append(seg_start + w * self.stride)

        self.window_starts = np.array(window_starts, dtype=np.int64)
        self.all_data = all_data
        self.data_stamp = np.zeros((len(all_data), 4), dtype=np.float32)

        print(f"  [Dataset_Haskins_Forecast] {n_speakers} speakers, "
              f"{len(sentences)} total sentences, "
              f"{n_groups} sentence groups, "
              f"{all_data.shape[1]} variates")
        print(f"  [Dataset_Haskins_Forecast] split by sentence_group: "
              f"train={len(train_groups)}g/{len(train_sents)}s, "
              f"val={len(val_groups)}g/{len(val_sents)}s, "
              f"test={len(test_groups)}g/{len(test_sents)}s")
        print(f"  [Dataset_Haskins_Forecast] {self.flag}: {len(split_sents)} sentences, "
              f"{len(self.window_starts)} windows "
              f"(stride={self.stride}, seq_len={self.seq_len}, pred_len={self.pred_len}, skipped={skipped})")

    def __getitem__(self, index):
        start = self.window_starts[index]
        s_end = start + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.all_data[start:s_end]
        seq_y = self.all_data[r_begin:r_end]
        seq_x_mark = self.data_stamp[start:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.window_starts)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
