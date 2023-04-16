
from IceCube.Essential import *


class MLP(nn.Sequential):
    def __init__(self, feats):
        layers = []
        for i in range(1, len(feats)):
            layers.append(nn.Linear(feats[i - 1], feats[i]))
            layers.append(nn.LeakyReLU())
        super().__init__(*layers)


class Model(pl.LightningModule):
    def __init__(
        self, max_lr=1e-3,
        num_warmup_step=1_000,
        remaining_step=1_000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conv0 = EdgeConv(MLP([17 * 2, 128, 256]), aggr="add")
        self.conv1 = EdgeConv(MLP([512, 336, 256]), aggr="add")
        self.conv2 = EdgeConv(MLP([512, 336, 256]), aggr="add")
        self.conv3 = EdgeConv(MLP([512, 336, 256]), aggr="add")
        self.post = MLP([1024 + 17, 336, 256])
        self.readout = MLP([768, 128])
        self.pred = nn.Linear(128, 3)

    def forward(self, data: Batch):
        vert_feat = data.x
        batch = data.batch

        # x, y, z, t, c, a
        # 0  1  2  3  4  5
        vert_feat[:, 0] /= 500.0  # x
        vert_feat[:, 1] /= 500.0  # y
        vert_feat[:, 2] /= 500.0  # z
        vert_feat[:, 3] = (vert_feat[:, 3] - 1.0e04) / 3.0e4  # time
        vert_feat[:, 4] = torch.log10(vert_feat[:, 4]) / 3.0  # charge

        edge_index = knn_graph(vert_feat[:, :3], 8, batch)

        # Construct global features
        hx = homophily(edge_index, vert_feat[:, 0], batch).reshape(-1, 1)
        hy = homophily(edge_index, vert_feat[:, 1], batch).reshape(-1, 1)
        hz = homophily(edge_index, vert_feat[:, 2], batch).reshape(-1, 1)
        ht = homophily(edge_index, vert_feat[:, 3], batch).reshape(-1, 1)
        means = scatter_mean(vert_feat, batch, dim=0)
        n_p = torch.log10(data.n_pulses).reshape(-1, 1)
        global_feats = torch.cat(
            [means, hx, hy, hz, ht, n_p], dim=1)  # [B, 11]

        # Distribute global_feats to each vertex
        _, cnts = torch.unique_consecutive(batch, return_counts=True)
        global_feats = torch.repeat_interleave(global_feats, cnts, dim=0)
        vert_feat = torch.cat((vert_feat, global_feats), dim=1)

        # Convolutions
        feats = [vert_feat]
        # Conv 0
        vert_feat = self.conv0(vert_feat, edge_index)
        feats.append(vert_feat)
        # Conv 1
        edge_index = knn_graph(vert_feat[:, :3], k=8, batch=batch)
        vert_feat = self.conv1(vert_feat, edge_index)
        feats.append(vert_feat)
        # Conv 2
        edge_index = knn_graph(vert_feat[:, :3], k=8, batch=batch)
        vert_feat = self.conv2(vert_feat, edge_index)
        feats.append(vert_feat)
        # Conv 3
        edge_index = knn_graph(vert_feat[:, :3], k=8, batch=batch)
        vert_feat = self.conv3(vert_feat, edge_index)
        feats.append(vert_feat)

        # Postprocessing
        post_inp = torch.cat(feats, dim=1)
        post_out = self.post(post_inp)

        # Readout
        readout_inp = torch.cat(
            [
                scatter_min(post_out, batch, dim=0)[0],
                scatter_max(post_out, batch, dim=0)[0],
                scatter_mean(post_out, batch, dim=0),
            ],
            dim=1,
        )
        readout_out = self.readout(readout_inp)

        # Predict
        pred = self.pred(readout_out)
        kappa = pred.norm(dim=1, p=2) + 1e-8
        pred_x = pred[:, 0] / kappa
        pred_y = pred[:, 1] / kappa
        pred_z = pred[:, 2] / kappa
        pred = torch.stack([pred_x, pred_y, pred_z, kappa], dim=1)

        return pred

    def train_or_valid_step(self, data, prefix):
        pred_xyzk = self.forward(data)  # [B, 4]
        true_xyz = data.gt.view(-1, 3)  # [B, 3]
        loss = VonMisesFisher3DLoss()(pred_xyzk, true_xyz).mean()
        error = angular_error(pred_xyzk[:, :3], true_xyz).mean()
        self.log(f"loss-{prefix}", loss, batch_size=len(true_xyz),
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"error-{prefix}", error, batch_size=len(true_xyz),
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def training_step(self, data, _):
        return self.train_or_valid_step(data, "train")

    def validation_step(self, data, _):
        return self.train_or_valid_step(data, "valid")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.max_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, 1e-2, 1, self.hparams.num_warmup_step
                ),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, 1, 1e-3, self.hparams.remaining_step
                ),
            ],
            milestones=[self.hparams.num_warmup_step],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def collate_fn(x):
    return x[0]


class LogCMK(torch.autograd.Function):
    """MIT License.

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indiciated in [1812.04616] in spite of suggestion in
    Sec. 8.2 of this paper. The change has been validated through comparison with
    exact calculations for `m=2` and `m=3` and found to yield the correct results.
    """

    @staticmethod
    def forward(ctx, m, kappa):  # pylint: disable=invalid-name,arguments-differ
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())).to(
            kappa.device
        )
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=invalid-name,arguments-differ
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(m / 2.0, kappa)) /
            (scipy.special.iv(m / 2.0 - 1, kappa))
        )
        return (
            None,
            grad_output *
            torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )


class VonMisesFisher3DLoss(nn.Module):
    """General class for calculating von Mises-Fisher loss.

    Requires implementation for specific dimension `m` in which the target and
    prediction vectors need to be prepared.
    """

    @classmethod
    def log_cmk_exact(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
        return LogCMK.apply(m, kappa)

    @classmethod
    def log_cmk_approx(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.

        [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
        """
        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa**2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    def log_cmk(
        cls, m: int, kappa: Tensor, kappa_switch: float = 100.0
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.

        Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
        precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
        this method automatically switches between the two at `kappa_switch`,
        ensuring continuity at this point.
        """
        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch

        # Ensure continuity at `kappa_switch`
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
            m, kappa_switch
        )
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    def _evaluate(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a vector in D dimensons.

        This loss utilises the von Mises-Fisher distribution, which is a
        probability distribution on the (D - 1) sphere in D-dimensional space.

        Args:
            prediction: Predicted vector, of shape [batch_size, D].
            target: Target unit vector, of shape [batch_size, D].

        Returns:
            Elementwise von Mises-Fisher loss terms.
        """
        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk(m, k) - dotprod
        return elements

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a direction in the 3D.

        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of `direction` and last column
                is an estimate of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        kappa = prediction[:, 3]
        p = kappa.unsqueeze(1) * prediction[:, [0, 1, 2]]
        return self._evaluate(p, target)
