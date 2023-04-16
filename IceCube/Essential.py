
# region imports

# basic imports
import time
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

# torch imports
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split, IterableDataset
from torch.distributions import Gamma, Normal
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import homophily
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv, knn_graph
import pytorch_lightning as pl
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

# other imports
import random
import copy
import yaml
from tqdm import tqdm
import pyarrow.parquet as pq
from math import floor

from IceCube.Helper import *

dtype = {
    "batch_id": "int16",
    "event_id": "int64",
}
# endregion

# =============================================================================
# training parameters
# =============================================================================
EPOCHS = 100
BATCH_SIZE = 250
EVENTS_PER_FILE = 200_000
# =============================================================================
# to include files in training
# =============================================================================

BATCHES_TRAIN = list(range(101, 601))
BATCHES_TUNE = list(range(101, 660, 2))  # tune-1
# BATCHES_TUNE = list(range(101, 660)) # tune-2
BATCHES_EVENTVAR = list(range(101, 601))
BATCHES_VALID = list(range(61, 80))
BATCHES_FIT = list(range(81, 86))
BATCHES_TEST = list(range(1, 101))

# basic settings
LOGGER = get_logger("IceCube", "DEBUG")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch_geometric.seed_everything(SEED)
pl.seed_everything(SEED)

# torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_start_method("spawn", force=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"using {DEVICE}")

# matplotlib
plt.set_loglevel("info")

# paths
BASE_PATH = "/root/autodl-tmp/kaggle/"
MODEL_PATH = os.path.join(BASE_PATH, "input", "ice-cube-model")
OUTPUT_PATH = os.path.join(BASE_PATH, "working")
PATH = os.path.join(BASE_PATH, "icecube-neutrinos-in-deep-ice")
PRED_PATH = os.path.join(BASE_PATH, "working", "prediction")
FILES_TRAIN, BATCHES_TRAIN = walk_dir(
    os.path.join(PATH, "train"), BATCHES_TRAIN)
FILES_TEST, BATCHES_TEST = walk_dir(os.path.join(PATH, "train"), BATCHES_TEST)
FILE_TRAIN_META = os.path.join(PATH, "train_meta.parquet")
FILE_TEST_META = os.path.join(PATH, "train_meta.parquet")
FILE_SENSOR_GEO = os.path.join(PATH, "sensor_geometry.csv")
FILE_GNN = os.path.join(MODEL_PATH, "finetuned.ckpt")
FILE_BDT = os.path.join(MODEL_PATH, "BDT_clf.Baseline.0414.sklearn")
LOGGER.info(f"{len(FILES_TRAIN)} files for training")
LOGGER.info(f"{len(FILES_TEST)} files for testing")
memory_check(LOGGER)


# Column names
col_xyzk = ["x", "y", "z", "kappa"]
col_angles = ["azimuth", "zenith"]
col_norm_vec = ["ex", "ey", "ez"]
col_dt = ["dt_15", "dt_50", "dt_85"]
col_qv = ["qx", "qy", "qz"]
col_xyzt = [
    "x0", "y0", "z0", "t0",
    "x1", "y1", "z1", "t1",
    "x2", "y2", "z2", "t2",
    "x3", "y3", "z3", "t3", ]
col_unique = ["uniq_x", "uniq_y", "uniq_z"]
col_glob_feat = ["hits", "error", "sumq", "meanq", "bratio"]
col_extra = col_norm_vec + col_dt + col_qv + \
    col_xyzt + col_unique + col_glob_feat

col_eventCat = ["error", "hits", "sumq", "qz",
                "dt_15", "dt_50", "dt_85", "ez", "uniq_x"]

# sensor geometry


def prepare_sensors(scale=None):
    sensors = pd.read_csv(FILE_SENSOR_GEO).astype({
        "sensor_id": np.int16,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    })

    if scale is not None and isinstance(scale, float):
        sensors["x"] *= scale
        sensors["y"] *= scale
        sensors["z"] *= scale

    return sensors


def angle_to_xyz(angles_b):
    az, zen = angles_b.t()
    x = torch.cos(az) * torch.sin(zen)
    y = torch.sin(az) * torch.sin(zen)
    z = torch.cos(zen)
    return torch.stack([x, y, z], dim=1)


def xyz_to_angle(xyz_b):
    x, y, z = xyz_b.t()
    az = torch.arccos(x / torch.sqrt(x**2 + y**2)) * torch.sign(y)
    zen = torch.arccos(z / torch.sqrt(x**2 + y**2 + z**2))
    return torch.stack([az, zen], dim=1)


def angular_error(xyz_pred_b, xyz_true_b):
    return torch.arccos(torch.clip_(torch.sum(xyz_pred_b * xyz_true_b, dim=1), -1, 1))


def angles2vector(df):
    df["nx"] = np.sin(df.zenith) * np.cos(df.azimuth)
    df["ny"] = np.sin(df.zenith) * np.sin(df.azimuth)
    df["nz"] = np.cos(df.zenith)
    return df


def vector2angles(n, eps=1e-8):
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + eps)
    azimuth = np.arctan2(n[:, 1],  n[:, 0])
    azimuth[azimuth < 0] += 2*np.pi
    zenith = np.arccos(n[:, 2].clip(-1, 1))
    return azimuth, zenith


def series2tensor(series, set_device=None):
    ret = torch.from_numpy(series.values).float()
    if set_device is not None:
        return ret.to(DEVICE)
    return ret


def angle_errors(n1, n2, eps=1e-8):
    n1 = n1 / (np.linalg.norm(n1, axis=1, keepdims=True) + eps)
    n2 = n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + eps)

    cos = (n1*n2).sum(axis=1)
    angle_err = np.arccos(cos.clip(-1, 1))

    r1 = n1[:, 0]*n1[:, 0] + n1[:, 1]*n1[:, 1]
    r2 = n2[:, 0]*n2[:, 0] + n2[:, 1]*n2[:, 1]
    cosX = (n1[:, 0]*n2[:, 0] + n1[:, 1]*n2[:, 1]) / (np.sqrt(r1*r2) + eps)
    azimuth_err = np.arccos(cosX.clip(-1, 1))

    zeros = r1 < eps
    azimuth_err[zeros] = np.random.random((len(n1[zeros]),))*np.pi

    zenith1 = np.arccos(n1[:, 2].clip(-1, 1))
    zenith2 = np.arccos(n2[:, 2].clip(-1, 1))
    zenith_err = np.abs(zenith2 - zenith1)

    return angle_err, azimuth_err, zenith_err


def get_target_angles(batches):
    res = None
    file = pq.ParquetFile(FILE_TRAIN_META)
    tmp = set(copy.copy(batches))
    for b in file.iter_batches(batch_size=EVENTS_PER_FILE, columns=["event_id", "batch_id", "azimuth", "zenith"]):
        if len(tmp) == 0:
            break
        true_df = b.to_pandas()
        batch_id = true_df.batch_id[0]
        if batch_id in tmp:
            true_df.event_id = true_df.event_id.astype(np.int64)
            true_df.azimuth = true_df.azimuth.astype(np.float32)
            true_df.zenith = true_df.zenith.astype(np.float32)
            true_df = true_df[["event_id", "batch_id", "azimuth", "zenith"]]
            res = true_df if res is None else pd.concat((res, true_df))
            tmp.remove(batch_id)
    return res


def get_reco_angles(batches):
    res = None
    for b in batches:
        file_name = f"pred_{b}.parquet"
        reco_df = pd.read_parquet(os.path.join(PRED_PATH, file_name))
        reco_df["azimuth"] = np.remainder(reco_df["azimuth"], 2 * np.pi)
        res = reco_df if res is None else pd.concat((res, reco_df))
    return res


def solve_linear(xw, yw, zw, xxw, yyw, xyw, yzw, zxw):
    A = torch.tensor([
        [xxw, xyw, xw],
        [xyw, yyw, yw],
        [xw,  yw,  1],
    ])
    b = torch.tensor([
        zxw, yzw, zw
    ])
    try:
        coeff = torch.linalg.solve(A, b)
        return coeff
    except Exception:
        LOGGER.debug("linear system not solvable")
        return torch.zeros((3, ))


# list of variables
def feature_extraction(df, fun=None, eps=1e-8):
    # sort by time
    df.sort_values(["time"], inplace=True)

    t = series2tensor(df.time)
    c = series2tensor(df.charge)
    x = series2tensor(df.x)
    y = series2tensor(df.y)
    z = series2tensor(df.z)

    # hits
    hits = t.numel()

    # weighted values
    Sx = torch.sum(x)
    Sxx = torch.sum(x*x)
    Sxy = torch.sum(x*y)
    Sy = torch.sum(y)
    Syy = torch.sum(y*y)
    Syz = torch.sum(y*z)
    Sz = torch.sum(z)
    Szx = torch.sum(z*x)

    # error of plane estimate
    coeff = solve_linear(Sx, Sy, Sz, Sxx, Syy, Sxy, Syz, Szx)
    error = torch.sum((z - coeff[0] * x - coeff[1] * y - coeff[2]))
    # error
    error = torch.square(error * 1e3)

    # plane norm vector
    norm_vec = torch.tensor([coeff[0], coeff[1], -1], dtype=torch.float)
    # norm_vec -> (3, )
    norm_vec /= torch.sqrt(coeff[0]**2 + coeff[1]**2 + 1)

    # delta t -> median time
    dt = torch.quantile(t, torch.tensor(
        [0.15, 0.50, 0.85], dtype=torch.float))                 # dt -> (3, )

    # charge centre (vector)
    # sumq
    sumq = torch.sum(c)
    # meanq
    meanq = sumq / hits
    qv = torch.tensor([torch.sum(x*c), torch.sum(y*c),
                      torch.sum(z*c)], dtype=torch.float)
    # qv -> (3, )
    qv /= sumq

    # bright sensor ratio
    # bratio
    bratio = c[c > 5 * meanq].numel() / hits

    # grouping by time (remember to sort by time)
    # xyzt -> (16, )
    n_groups = 4

    if hits > n_groups:
        sec_len = floor(hits / n_groups)
        remain_len = hits - (n_groups - 1) * sec_len
        xyzt = series2tensor(df[["x", "y", "z", "time"]])
        xyzt = torch.split(xyzt, [sec_len, sec_len, sec_len, remain_len])
        xyzt = torch.concat([xx.mean(axis=0) for xx in xyzt])
    else:
        xyzt = torch.zeros(n_groups * 4)
        _xxxx = list()
        for i in range(hits):
            _xxxx.append(x[i])
            _xxxx.append(y[i])
            _xxxx.append(z[i])
            _xxxx.append(t[i])
        xyzt[: hits * 4] = torch.tensor(_xxxx, dtype=torch.float)

    # unique xyz
    unique = torch.tensor([_x.unique().numel() for _x in [
                          x, y, z]], dtype=torch.float)         # unique -> (3, )

    # global features
    glob_feat = torch.tensor(
        [hits, error, sumq, meanq, bratio, ], dtype=torch.float)

    return torch.concat([norm_vec, dt, qv, xyzt, unique, glob_feat]).unsqueeze(0)


def prepare_feature(df):
    df = df.reset_index(drop=True)
    # remove auxiliary
    df = df[~df.auxiliary]
    df.x *= 1e-3
    df.y *= 1e-3
    df.z *= 1e-3
    df.time -= np.min(df.time)
    return df[["time", "charge", "x", "y", "z"]]


# Dataset
class IceCube(IterableDataset):
    """
    smear: statistical fluctuation
    smear_rate: apply smearing on one of N samples
    event_category: once train on certain category, 0: hard, 1: easy, None: train on all
    cat_model: model that provide the categorization
    """

    def __init__(
        self, parquet_dir, meta_dir, chunk_ids,
        batch_size=200, max_pulses=200, shuffle=False, extra=False,
        smear=False, smear_rate=2, event_category=None, cat_model=None
    ):
        self.parquet_dir = parquet_dir
        self.meta_dir = meta_dir
        self.chunk_ids = chunk_ids
        self.batch_size = batch_size
        self.max_pulses = max_pulses
        self.shuffle = shuffle
        self.extra = extra
        self.smear = smear
        self.smear_rate = smear_rate
        self.event_category = event_category
        self.cat_model = cat_model
        self.eventCat_var_list = [col_extra.index(nm) for nm in col_eventCat]

        if self.shuffle:
            random.shuffle(self.chunk_ids)

    def __iter__(self):
        # Handle num_workers > 1 and multi-gpu
        is_dist = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_dist else 1
        rank_id = torch.distributed.get_rank() if is_dist else 0

        info = torch.utils.data.get_worker_info()
        num_worker = info.num_workers if info else 1
        worker_id = info.id if info else 0

        num_replica = world_size * num_worker
        offset = rank_id * num_worker + worker_id
        chunk_ids = self.chunk_ids[offset::num_replica]

        # Sensor data
        sensor = prepare_sensors()

        # Read each chunk and meta iteratively into memory and build mini-batch
        for c, chunk_id in enumerate(chunk_ids):
            data = pd.read_parquet(os.path.join(
                self.parquet_dir, f"batch_{chunk_id}.parquet"))
            meta = pd.read_parquet(os.path.join(
                self.meta_dir, f"meta_{chunk_id}.parquet"))

            angles = meta[["azimuth", "zenith"]].values
            angles = torch.from_numpy(angles).float()
            xyzs = angle_to_xyz(angles)
            meta = {eid: xyz for eid, xyz in zip(
                meta["event_id"].tolist(), xyzs)}

            # Take all event_ids and split them into batches
            eids = list(meta.keys())
            if self.shuffle:
                random.shuffle(eids)
            eids_batches = [
                eids[i: i + self.batch_size]
                for i in range(0, len(eids), self.batch_size)
            ]

            for batch_eids in eids_batches:
                batch = []

                # For each sample, extract features
                for eid in batch_eids:
                    df = data.loc[eid]
                    df = pd.merge(df, sensor, on="sensor_id")
                    # sampling of pulses if number exceeds maximum
                    if len(df) > self.max_pulses:
                        df_pass = df[~df.auxiliary]
                        df_fail = df[df.auxiliary]
                        if len(df_pass) >= self.max_pulses:
                            df = df_pass.sample(self.max_pulses)
                        else:
                            df_fail = df_fail.sample(
                                self.max_pulses - len(df_pass))
                            df = pd.concat([df_fail, df_pass])

                    df.sort_values(["time"], inplace=True)

                    t = series2tensor(df.time)
                    c = series2tensor(df.charge)
                    a = series2tensor(df.auxiliary)
                    x = series2tensor(df.x)
                    y = series2tensor(df.y)
                    z = series2tensor(df.z)

                    # smearing
                    if self.smear and eid % self.smear_rate == 0:  # smear half of the dataset
                        # x, y, z are fixed ...
                        dist_normal = Normal(0, 1.2)  # time resolution = 1.2ns
                        # poisson statistics of photoelectrics
                        dist_gamma = Gamma(c * 10, 10)
                        t += dist_normal.sample()
                        c = dist_gamma.sample()

                    feat = torch.stack([x, y, z, t, c, a], dim=1)

                    batch_data = Data(x=feat, gt=meta[eid],
                                      n_pulses=len(feat), eid=torch.tensor([eid]).long(),
                                      )

                    if self.extra:
                        feats = feature_extraction(prepare_feature(df))
                        setattr(batch_data, "extra_feat", feats)

                    if self.event_category is not None:
                        X = feats[:, self.eventCat_var_list]
                        category = 0 if self.cat_model.predict(
                            X.numpy()) else 1

                    if self.event_category is None or category == self.event_category:
                        batch.append(batch_data)

                yield Batch.from_data_list(batch)

            del data
            del meta
            gc.collect()
