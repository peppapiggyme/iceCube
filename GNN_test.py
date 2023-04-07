
from IceCube.Essential import *
from IceCube.Model import *
import pdb


def produce_prediction(model, parquet_dir, meta_dir, batch_num=1):

    output_name = f"pred_{batch_num}.parquet"
    output_file = os.path.join(PRED_PATH, output_name)

    test_set = IceCube(
        parquet_dir, meta_dir, [batch_num], batch_size=BATCH_SIZE, use_fit=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=1,
    )

    pred = None
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            pred_xyzk = model(data.to(DEVICE))
            angles = np.concatenate([
               # +--------------------+------------------------------------+------------------+
               # |   x, y, z, kappa   |           azimuth, zenith          |    fit outputs   |
               # +--------------------+------------------------------------+------------------+
                pred_xyzk[:, :3].cpu(), xyz_to_angle(pred_xyzk[:, :3]).cpu(), data.plane.cpu()
               # +--------------------+------------------------------------+------------------+
            ], axis=1)
            pred = angles if pred is None else np.concatenate([pred, angles]) 
    
    res = pd.DataFrame(pred, 
        columns=["x", "y", "z", "azimuth", "zenith",
                 "ex", "ey", "ez", "fit_error", "hits", "sumw", "sumc", "sumt", "dt", 
                 "std_x", "std_z", "std_t", "unique_x", "unique_z"])
    res.to_parquet(output_file)
    print(res)


class ModelFormat:
    pth = 0
    ckpt = 1


if __name__ == "__main__":

    format = ModelFormat.ckpt

    file_pth = "official-pretrained.pth"
    file_ckpt = "pretrained-800000.ckpt"

    parquet_dir = os.path.join(PATH, "train")
    meta_dir = os.path.join(PATH, "train_meta")

    if format == ModelFormat.pth:
        model = Model()
        weights = torch.load(os.path.join(MODEL_PATH, file_pth))
        model.load_state_dict(weights)
        LOGGER.info(f"loaded {file_pth}")
    elif format == ModelFormat.ckpt:
        model = Model.load_from_checkpoint(os.path.join(MODEL_PATH, file_ckpt))
        LOGGER.info(f"loaded {file_ckpt}")

    model.eval()
    model.freeze()
    model.to(DEVICE)

    for i in BATCHES_TEST:
        produce_prediction(model, parquet_dir, meta_dir, batch_num=i)
