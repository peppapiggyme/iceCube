
#region imports

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
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import homophily
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv, knn_graph
import pytorch_lightning as pl
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

# other imports
import random, copy, yaml
from tqdm import tqdm
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d

from IceCube.Helper import *

dtype = {
    "batch_id": "int16",
    "event_id": "int64",
}
#endregion

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
BATCHES_VALID = list(range(61, 80))
# BATCHES_FIT = list(range(1, 51, 10))
BATCHES_FIT = [1]
BATCHES_TEST = list(range(1, 21))

# basic settings
LOGGER = get_logger("GNN", "DEBUG")
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
MODEL_PATH = BASE_PATH + "../models/"
PATH = os.path.join(BASE_PATH, "icecube-neutrinos-in-deep-ice")
PRED_PATH = os.path.join(BASE_PATH, "working", "prediction")
FILES_TRAIN, BATCHES_TRAIN = walk_dir(os.path.join(PATH, "train"), BATCHES_TRAIN)
FILES_TEST, BATCHES_TEST = walk_dir(os.path.join(PATH, "train"), BATCHES_TEST)
FILE_TRAIN_META = os.path.join(PATH, "train_meta.parquet")
FILE_TEST_META = os.path.join(PATH, "train_meta.parquet")
FILE_SENSOR_GEO = os.path.join(PATH, "sensor_geometry.csv")
FILE_ICE_TRANS = os.path.join(PATH, "ice_transparency_info.csv")
FILE_PARAM = os.path.join(PATH, "parameters.yaml")
LOGGER.info(f"{len(FILES_TRAIN)} files for training")
LOGGER.info(f"{len(FILES_TEST)} files for testing")
memory_check(LOGGER)

# BEST_FIT
BEST_FIT_VALUES = None
with open(FILE_PARAM, "r") as f:
    BEST_FIT_VALUES = yaml.full_load(f)
LOGGER.info(f"best fit values:\n{BEST_FIT_VALUES}")


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


def Rx(theta):
    return np.array([
        [1,  0,             0            ],
        [0,  np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta), np.cos(theta)]
    ])


def Ry(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def Rz(theta):
    return np.array([
        [ np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [ 0,             0,             1]
    ])


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
    azimuth = np.arctan2( n[:,1],  n[:,0])    
    azimuth[azimuth < 0] += 2*np.pi
    zenith = np.arccos( n[:,2].clip(-1,1) )                                
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
    angle_err = np.arccos( cos.clip(-1,1) )
        
    r1   =  n1[:,0]*n1[:,0] + n1[:,1]*n1[:,1] 
    r2   =  n2[:,0]*n2[:,0] + n2[:,1]*n2[:,1]
    cosX = (n1[:,0]*n2[:,0] + n1[:,1]*n2[:,1]) / (np.sqrt(r1*r2) + eps)    
    azimuth_err = np.arccos( cosX.clip(-1,1) )
                                
    zeros = r1 < eps
    azimuth_err[zeros] = np.random.random((len(n1[zeros]),))*np.pi
    
    zenith1  = np.arccos( n1[:,2].clip(-1,1) )
    zenith2  = np.arccos( n2[:,2].clip(-1,1) )
    zenith_err = np.abs(zenith2 - zenith1)    
        
    return angle_err, azimuth_err, zenith_err


def get_target_angles(batches):
    res = None
    file = pq.ParquetFile(FILE_TRAIN_META)
    tmp = copy.copy(batches)
    for b in file.iter_batches(batch_size=EVENTS_PER_FILE, columns=["event_id","batch_id","azimuth","zenith"]):    
        if len(tmp) == 0:
            break
        true_df = b.to_pandas()
        batch_id = true_df.batch_id[0]
        if batch_id in tmp:      
            true_df.event_id= true_df.event_id.astype(np.int64)      
            true_df.azimuth = true_df.azimuth.astype(np.float32)
            true_df.zenith  = true_df.zenith.astype(np.float32)    
            true_df = true_df[["event_id", "batch_id", "azimuth", "zenith"]]
            res =  true_df if res is None else pd.concat((res, true_df))            
            tmp.remove(batch_id)
    return res


def get_reco_angles(batches):
    res = None
    for b in batches:
        file_name = f"pred_{b}.parquet"
        reco_df = pd.read_parquet(os.path.join(PRED_PATH, file_name))
        reco_df["azimuth"] = np.remainder(reco_df["azimuth"], 2 * np.pi)
        res =  reco_df if res is None else pd.concat((res, reco_df))
    return res            


def prepare_batch(df, sensor):
    df["event_id"] = df.index.astype(np.int64)    
    df = df.reset_index(drop=True)

    # remove auxiliary
    df = df[~df.auxiliary]

    df.charge = df.charge.astype(np.float32)
    df.charge = np.clip(df.charge, 0, 4)
    times = df.groupby("event_id").agg(
        t_min = ("time", np.min),
    )
    
    df = df.merge(times, on="event_id")
    df.time = ((df.time - df.t_min) * 0.299792458e-3).astype(np.float32)
    df = pd.merge(df, sensor, on="sensor_id")

    df["qz"] = df.charge * df.z

    centre = df.groupby(["event_id", "x", "y"]).agg(
        qsum = ("charge", np.sum),
        qzsum = ("qz", np.sum),
    )

    centre["z_avg"] = centre.qzsum / centre.qsum
    df = pd.merge(df, centre[["z_avg"]], on=["event_id", "x", "y"])

    return df


def solve_linear(xw, yw, zw, xxw, yyw, zzw, xyw, yzw, zxw):
    A = torch.tensor([
        [xxw, xyw, xw],
        [xyw, yyw, yw],
        [xw,  yw,  1 ],
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


def plane_fit(df, k=0, kt=0, kq=0, fun=None, eps=1e-8):
    z_avg = series2tensor(df.z_avg)
    t = series2tensor(df.time)
    c = series2tensor(df.charge)
    x = series2tensor(df.x)
    y = series2tensor(df.y)
    z = series2tensor(df.z)

    # weighted by ...
    w = torch.exp(-k * torch.square(z - z_avg)) \
        * torch.exp(-kt * t) \
        * torch.pow(c, kq)

    # weighted values
    xw = (x*w); xxw = (x*x*w); xyw = (x*y*w)
    yw = (y*w); yyw = (y*y*w); yzw = (y*z*w)
    zw = (z*w); zzw = (z*z*w); zxw = (z*x*w)  

    xw = torch.sum(xw); xxw = torch.sum(xxw); xyw = torch.sum(xyw) 
    yw = torch.sum(yw); yyw = torch.sum(yyw); yzw = torch.sum(yzw) 
    zw = torch.sum(zw); zzw = torch.sum(zzw); zxw = torch.sum(zxw) 
    sumw = torch.sum(w); sumc = torch.sum(w*c); dt = torch.median(t)

    sumw += eps
    xw /= sumw; xxw /= sumw; xyw /= sumw
    yw /= sumw; yyw /= sumw; yzw /= sumw
    zw /= sumw; zzw /= sumw; zxw /= sumw

    coeff = solve_linear(xw, yw, zw, xxw, yyw, zzw, xyw, yzw, zxw)
    error = torch.sum((z - coeff[0] * x - coeff[1] * y - coeff[2]))
    error *= 1e3
    hits = w.shape[0]
    unique_x = torch.unique(x).shape[0]
    unique_y = torch.unique(y).shape[0]
    unique_z = torch.unique(z).shape[0]

    ret = torch.tensor([[coeff[0], coeff[1], -1, torch.square(error), hits, sumc, dt, unique_x, unique_y, unique_z]])
    ret[:, :3] /= torch.sqrt(coeff[0]**2 + coeff[1]**2 + 1)

    return ret


def prepare_df_for_plane(df):
    df = df.reset_index(drop=True)

    # remove auxiliary
    df = df[~df.auxiliary]

    df.charge = df.charge.astype(np.float32)
    df.charge = np.clip(df.charge, 0, 4)
    t_min = np.min(df.time)
    df.time = ((df.time - t_min) * 0.299792458e-3).astype(np.float32)
    df.x *= 1e-3; df.y *= 1e-3; df.z *= 1e-3
    
    df["qz"] = df.charge * df.z
    centre = df.groupby(["x", "y"]).agg(
        qsum = ("charge", np.sum),
        qzsum = ("qz", np.sum),
    )

    centre["z_avg"] = centre.qzsum / centre.qsum
    df = pd.merge(df, centre[["z_avg"]], on=["x", "y"])

    return df[["z_avg", "time", "charge", "x", "y", "z"]]


# Dataset
class IceCube(IterableDataset):
    def __init__(
        self, parquet_dir, meta_dir, chunk_ids,
        batch_size=200, max_pulses=200, shuffle=False, use_fit=False
    ):
        self.parquet_dir = parquet_dir
        self.meta_dir = meta_dir
        self.chunk_ids = chunk_ids
        self.batch_size = batch_size
        self.max_pulses = max_pulses
        self.shuffle = shuffle
        self.use_fit = use_fit

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
            data = pd.read_parquet(os.path.join(self.parquet_dir, f"batch_{chunk_id}.parquet"))
            meta = pd.read_parquet(os.path.join(self.meta_dir, f"meta_{chunk_id}.parquet"))

            angles = meta[["azimuth", "zenith"]].values
            angles = torch.from_numpy(angles).float()
            xyzs = angle_to_xyz(angles)
            meta = {eid: xyz for eid, xyz in zip(meta["event_id"].tolist(), xyzs)}

            # Take all event_ids and split them into batches
            eids = list(meta.keys())
            if self.shuffle:
                random.shuffle(eids)
            eids_batches = [
                eids[i : i + self.batch_size]
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
                            df_fail = df_fail.sample(self.max_pulses - len(df_pass))
                            df = pd.concat([df_fail, df_pass])

                    df.sort_values(["time"], inplace=True)

                    t = series2tensor(df.time)
                    c = series2tensor(df.charge)
                    a = series2tensor(df.auxiliary)
                    x = series2tensor(df.x)
                    y = series2tensor(df.y)
                    z = series2tensor(df.z)
                    feat = torch.stack([x, y, z, t, c, a], dim=1)

                    batch_data = Data(x=feat, gt=meta[eid],
                        n_pulses=len(feat), eid=torch.tensor([eid]).long(),
                    )

                    if self.use_fit:
                        coeff = plane_fit(prepare_df_for_plane(df), **BEST_FIT_VALUES)
                        setattr(batch_data, "plane", coeff)

                    batch.append(batch_data)

                yield Batch.from_data_list(batch)

            del data
            del meta
            gc.collect()

