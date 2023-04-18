
from IceCube.Essential import *
from IceCube.Model import *
import pickle
import argparse
import pdb


if __name__ == "__main__":

    # optimise learning rate
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', metavar='LR', type=float,
                        help='maximum learning rate')
    args = parser.parse_args()

    # Config
    step_per_epoch = int(len(BATCHES_EVENTVAR) * EVENTS_PER_FILE / BATCH_SIZE)
    num_total_step = EPOCHS * step_per_epoch
    LOGGER.info(f"Total steps = {num_total_step}")
    num_warmup_step = int(step_per_epoch * 0.25)
    remaining_step = int(step_per_epoch * 5.75)

    parquet_dir = os.path.join(PATH, "train")
    meta_dir = os.path.join(PATH, "train_meta")

    log_dir = "/root/autodl-tmp/logs/"

    model = Model(
        max_lr=args.lr,
        num_warmup_step=num_warmup_step,
        remaining_step=remaining_step,
    )

    ckpt_file = os.path.join(MODEL_PATH, "finetuned-v2.ckpt")
    state_dict = torch.load(ckpt_file)["state_dict"]
    model.load_state_dict(state_dict)

    clf = pickle.load(
        open(os.path.join(MODEL_PATH, 'EventCat_clf.Tree.10.sklearn'), 'rb'))

    # randomly smear time and charge in training data
    train_set = IceCube(
        parquet_dir, meta_dir, BATCHES_EVENTVAR, batch_size=BATCH_SIZE, shuffle=True, extra=True,
        smear=False, event_category=0, cat_model=clf,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        num_workers=32,
    )

    # not for validation data apparently
    valid_set = IceCube(
        parquet_dir, meta_dir, BATCHES_VALID, batch_size=BATCH_SIZE, extra=True,
        event_category=0, cat_model=clf,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        num_workers=16,
    )

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        logger=pl.loggers.CSVLogger(log_dir),
        accelerator="gpu",
        devices=2,
        max_steps=num_total_step,
        log_every_n_steps=50 * EVENTS_PER_FILE / BATCH_SIZE,  # 50 files
        val_check_interval=50 * EVENTS_PER_FILE / BATCH_SIZE,  # 50 files
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelSummary(),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(log_dir, save_top_k=-1),
        ],
    )

    trainer.fit(model, train_loader, valid_loader)
