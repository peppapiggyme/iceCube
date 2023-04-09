
from IceCube.Essential import *
from IceCube.Model import *
from scipy.optimize import differential_evolution, minimize, Bounds
import pdb, yaml


def svd_agg(x):
    w = x[["w"]].values
    x = x[["x", "y", "z"]].values
    x = x * w
    n = x.shape[0]
    if n > 3:
        x = np.split(x, [int(np.floor(n/3)), int(np.floor(2*n/3))])
        x = np.concatenate([xx.mean(axis=0)[np.newaxis, :] for xx in x])

    _, S, V = np.linalg.svd(x, full_matrices=False)
    if V.shape == (0, 3):
        V = np.zeros((3, 3))
    elif V.shape == (1, 3):
        V = np.concatenate((V, V, V))
    elif V.shape == (2, 3):
        tmp = np.zeros((3, 3))
        tmp[:2, :] = V
        tmp[2, :] = 2 * V[1, :] - V[0, :]
        V = tmp
    result = pd.DataFrame({"x": [V[0,0]], "y": [V[1,0]], "z": [V[2,0]]})
    global i
    i = i + 1
    if i % 10000 == 0: 
        memory_check(LOGGER)
        LOGGER.info(f"processed {i} samples")
    return result


def solve_linear(point):
    A = np.array([
        [point.xxw, point.xyw, point.xw],
        [point.xyw, point.yyw, point.yw],
        [point.xw,  point.yw,  1       ],
    ])
    b = np.array([
        point.zxw, point.yzw, point.zw
    ])
    try:
        coeff = np.linalg.solve(A, b)
        return coeff[np.newaxis,:]
    except Exception:
        LOGGER.debug("linear system not solvable")
        return np.zeros((1, 3))


def plane_fit(df, k=0, kt=0, kq=0, eps=1e-8):
    # weighted by ...
    df["w"] = np.exp(-k * np.abs(df.z - df.z_avg)) \
        * np.exp(-kt * df.time)
    
    # weighted values
    df["xw"] = df.x * df.w; df["xxw"] = df.x * df.x * df.w; df["xyw"] = df.x * df.y * df.w
    df["yw"] = df.y * df.w; df["yyw"] = df.y * df.y * df.w; df["yzw"] = df.y * df.z * df.w
    df["zw"] = df.z * df.w; df["zzw"] = df.z * df.z * df.w; df["zxw"] = df.z * df.x * df.w  

    wtd = df.groupby("event_id").agg(
        xw = ("xw", np.sum), xxw = ("xxw", np.sum), xyw = ("xyw", np.sum), 
        yw = ("yw", np.sum), yyw = ("yyw", np.sum), yzw = ("yzw", np.sum), 
        zw = ("zw", np.sum), zzw = ("zzw", np.sum), zxw = ("zxw", np.sum), 
        sumw = ("w", np.sum)
    ).reset_index()

    # svd = df.groupby("event_id").apply(svd_agg)

    wtd.sumw += eps
    wtd.xw /= wtd.sumw; wtd.xxw /= wtd.sumw; wtd.xyw /= wtd.sumw
    wtd.yw /= wtd.sumw; wtd.yyw /= wtd.sumw; wtd.yzw /= wtd.sumw
    wtd.zw /= wtd.sumw; wtd.zzw /= wtd.sumw; wtd.zxw /= wtd.sumw

    res = None
    for _, row in wtd.iterrows():
        coeff = solve_linear(row)
        res = coeff if res is None else np.concatenate((res, coeff)) 

    return res


def func(x):
    coeff = plane_fit(pulses_df, x[0], x[1])
    LOGGER.debug(f"coeff\n{coeff}")
    norm = np.sqrt(coeff[:, 0]**2 + coeff[:, 1]**2 + 1)[:, np.newaxis]
    unit_vec = np.array([coeff[:, 0], coeff[:, 1], -1*np.ones(coeff[:, 1].shape)]).T
    unit_vec /= norm
    # Rot = np.linalg.multi_dot([Rz(x[4]), Ry(x[3]), Rx(x[2])])
    # unit_vec = np.einsum("mk,bm->bk", Rot, unit_vec)
    prod = np.abs(np.sum(unit_vec * n, axis=1)).mean()
    LOGGER.info(f"prod = {prod}, k = {x[0]:.6f}, kt = {x[1]:.6f}")

    # err, az, ze = angle_errors(svd.values, n)
    # LOGGER.info(f"svd result err = {err.mean()}")

    # xe = np.sum(svd.values * unit_vec, axis=1)
    # proj = svd - xe[:, np.newaxis] * unit_vec
    # proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)
    # err, az, ze = angle_errors(proj.values, n)
    # LOGGER.info(f"svd proj result err = {err.mean()}")

    return prod


def save_parameters(res, file_path):
    param = {
        "k"       : float(res.x[0]),
        "kt"      : float(res.x[1]),
        "fun"     : float(res.fun),
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
    del df; 
    print(pulses_df.head(2))

    true_df = get_target_angles(BATCHES_FIT)
    true_df = angles2vector(true_df)
    print(true_df.head(2))
    n = true_df[["nx","ny","nz"]].to_numpy()

    """ Step 1: find global minimum (roughly) """
    # +-------+------+--------+
    # | Bound |  k   |   kt   |
    # +-------+------+--------+
    bounds = [(0, 10), (0, 10)]
    # +-------+------+--------+
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

