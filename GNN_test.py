
from IceCube.Essential import *
from IceCube.Model import *
import pdb


def produce_prediction(model, parquet_dir, meta_dir, batch_num=1):

    output_name = f"pred_{batch_num}.parquet"
    output_file = os.path.join(BASE_PATH, "working", "prediction", output_name)

    test_set = IceCube(
        parquet_dir, meta_dir, [batch_num], batch_size=200, use_fit=True
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
        columns=["x", "y", "z", "azimuth", "zenith", "ex", "ey", "ez", "fit_error", "good_hits"])
    res.to_parquet(output_file)
    print(res)


if __name__ == "__main__":

    parquet_dir = os.path.join(PATH, "train")
    meta_dir = os.path.join(PATH, "train_meta")

    model = Model()
    weights = torch.load(os.path.join(MODEL_PATH, "official-pretrained.pth"))
    model.load_state_dict(weights)

    model.eval()
    model.to(DEVICE)

    for i in [1, 3, 4, 5]:
        produce_prediction(model, parquet_dir, meta_dir, batch_num=i)
