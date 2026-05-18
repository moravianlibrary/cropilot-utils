import torch
import numpy as np
from torch.utils.data import DataLoader
import streamlit as st

from base_model_trainer.network.rotate_dataset import PageAngleDataset
from base_model_trainer.network.rotate_network import AngleDegModel, load_checkpoint
from base_model_trainer.training.rotate_train import get_bbox_vectors, get_filepaths


def get_loader_for_degree(degree=0.0):
    test_ds = PageAngleDataset(
        image_paths=get_filepaths("datasets/yolo-all-batches-rotate/images/test"),
        image_bboxes=get_bbox_vectors(
            "datasets/yolo-all-batches-rotate/labels/test"
        ),
        is_train=True,
        image_size=640,
        angle_max=degree,
        angle_min=degree-0.1,
        aug_rotate_prob=1.0,
    )
    return DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

def plot_mae_per_degree(model):
    maes = {}
    with st.spinner("Plotting chart of MAE per degree..."):
        for degree in range(-10, 11):
            test_loader = get_loader_for_degree(degree)
            imgs, preds, trues = model.predict_angles(test_loader, device)
            mae = np.mean(np.abs(preds - trues))
            maes[degree] = mae

        st.write(maes)

        # plot the results
        st.subheader("MAE by Rotation Degree")
        st.bar_chart(maes)
        st.write("MAE across all degrees:", np.mean(list(maes.values())))

if __name__ == '__main__':
    device = torch.device("mps")
    model = AngleDegModel().to(device)
    checkpoint_path = "base_models/rotate-300e-best.pth"
    load_checkpoint(checkpoint_path, model, map_location=device)

    plot_mae_per_degree(model)