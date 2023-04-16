
from IceCube.Essential import *
from IceCube.Model import *
import pdb


if __name__ == "__main__":
    model = Model()

    # Verify with offical pretrained weight
    weights = torch.load(
        "/root/autodl-tmp/kaggle/input/dynedge-pretrained/dynedge_pretrained_batch_1_to_50/state_dict.pth")
    new_weights = dict()
    for k, v in weights.items():
        k = k.replace("_gnn._conv_layers.0", "conv0")
        k = k.replace("_gnn._conv_layers.1", "conv1")
        k = k.replace("_gnn._conv_layers.2", "conv2")
        k = k.replace("_gnn._conv_layers.3", "conv3")
        k = k.replace("_gnn._post_processing", "post")
        k = k.replace("_gnn._readout", "readout")
        k = k.replace("_tasks.0._affine", "pred")
        new_weights[k] = v
    print(model.load_state_dict(new_weights))

    torch.save(model.state_dict(), os.path.join(
        MODEL_PATH, "official-pretrained.pth"))
