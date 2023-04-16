
from IceCube.Essential import *
from IceCube.Model import *
from scipy.optimize import differential_evolution, minimize, Bounds
import pdb
import yaml


def prepare_batch(df, sensor):
    df["event_id"] = df.index.astype(np.int64)
    df = df.reset_index(drop=True)

    # remove auxiliary
    df = df[~df.auxiliary]

    df.charge = df.charge.astype(np.float32)
    df.charge = np.clip(df.charge, 0, 4)
    times = df.groupby("event_id").agg(
        t_min=("time", np.min),
    )

    df = df.merge(times, on="event_id")
    df.time = ((df.time - df.t_min) * 0.299792458e-3).astype(np.float32)
    df = pd.merge(df, sensor, on="sensor_id")

    return df


def solve_linear(point):
    A = np.array([
        [point.xxw, point.xyw, point.xw],
        [point.xyw, point.yyw, point.yw],
        [point.xw,  point.yw,  1],
    ])
    b = np.array([
        point.zxw, point.yzw, point.zw
    ])
    try:
        coeff = np.linalg.solve(A, b)
        return coeff[np.newaxis, :]
    except Exception:
        LOGGER.debug("linear system not solvable")
        return np.zeros((1, 3))


def plane_fit(df, kt=0, eps=1e-8):
    # weighted by ...
    df["w"] = np.exp(-kt * df.time)

    # weighted values
    df["xw"] = df.x * df.w
    df["xxw"] = df.x * df.x * df.w
    df["xyw"] = df.x * df.y * df.w
    df["yw"] = df.y * df.w
    df["yyw"] = df.y * df.y * df.w
    df["yzw"] = df.y * df.z * df.w
    df["zw"] = df.z * df.w
    df["zzw"] = df.z * df.z * df.w
    df["zxw"] = df.z * df.x * df.w

    wtd = df.groupby("event_id").agg(
        xw=("xw", np.sum), xxw=("xxw", np.sum), xyw=("xyw", np.sum),
        yw=("yw", np.sum), yyw=("yyw", np.sum), yzw=("yzw", np.sum),
        zw=("zw", np.sum), zzw=("zzw", np.sum), zxw=("zxw", np.sum),
        sumw=("w", np.sum)
    ).reset_index()

    wtd.sumw += eps
    wtd.xw /= wtd.sumw
    wtd.xxw /= wtd.sumw
    wtd.xyw /= wtd.sumw
    wtd.yw /= wtd.sumw
    wtd.yyw /= wtd.sumw
    wtd.yzw /= wtd.sumw
    wtd.zw /= wtd.sumw
    wtd.zzw /= wtd.sumw
    wtd.zxw /= wtd.sumw

    res = None
    for _, row in wtd.iterrows():
        coeff = solve_linear(row)
        res = coeff if res is None else np.concatenate((res, coeff))

    return res


def func(x):
    coeff = plane_fit(pulses_df, x[0])
    LOGGER.debug(f"coeff\n{coeff}")
    norm = np.sqrt(coeff[:, 0]**2 + coeff[:, 1]**2 + 1)[:, np.newaxis]
    unit_vec = np.array([coeff[:, 0], coeff[:, 1], -
                        1*np.ones(coeff[:, 1].shape)]).T
    unit_vec /= norm

    prod = np.abs(np.sum(unit_vec * n, axis=1)).mean()
    LOGGER.info(f"prod = {prod}, kt = {x[0]:.6f}")

    return prod


def save_parameters(res, file_path):
    param = {
        "kt": float(res.x[0]),
        "fun": float(res.fun),
    }

    with open(file_path, "w") as f:
        yaml.dump(param, f)

    print(f"ciao, saved to {file_path}")


if __name__ == "__main__":

    parquet_dir = os.path.join(PATH, "train")
    meta_dir = os.path.join(PATH, "train_meta")

    sensor = prepare_sensors(1e-3)
    print(sensor.head(2))

    pulses_df = None
    for i in BATCHES_FIT:
        df = pd.read_parquet(os.path.join(parquet_dir, f"batch_{i}.parquet"))
        df = prepare_batch(df, sensor)
        pulses_df = df if pulses_df is None else pd.concat((pulses_df, df))
    del df
    print(pulses_df.head(2))

    true_df = get_target_angles(BATCHES_FIT)
    true_df = angles2vector(true_df)
    print(true_df.head(2))
    n = true_df[["nx", "ny", "nz"]].to_numpy()

    """ Step 1: find global minimum (roughly) """
    bounds = [(0, 10)]
    res = differential_evolution(func, bounds, maxiter=100, popsize=12)
    LOGGER.info(res.x)
    LOGGER.info(res.fun)
    save_parameters(res, "../logs/parameters.yaml")

    """ Step 2: find global minimum (roughly) """
    # x0 = [BEST_FIT_VALUES['k'], BEST_FIT_VALUES['kt']]
    # res = minimize(func, x0, method="Nelder-Mead", tol=1e-6)
    # save_parameters(res, "../logs/parameters_local.yaml")

    """ Test the best-fit parameters """
    # x0 = [BEST_FIT_VALUES['k'], BEST_FIT_VALUES['kt'], BEST_FIT_VALUES['kq']]
    # x0 = [1, 1, 1]
    # func(x0)
