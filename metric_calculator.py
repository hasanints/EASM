# metric_calculator.py

from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, confusion_matrix
import numpy as np
import pandas as pd
import torch
import os

class MetricsCalculator:
    """
    Class to calculate and save metrics after training.
    """
    def __init__(self, config, checkpoint_dir):
        self.config = config
        self.checkpoint_dir = checkpoint_dir

    def _calc_metrics(self):
        """
        Calculate and save metrics after training for all folds.
        """
        n_folds = self.config["data_loader"]["args"]["num_folds"]
        all_outs = []
        all_trgs = []

        outs_list = []
        trgs_list = []
        save_dir = os.path.abspath(os.path.join(self.checkpoint_dir, os.pardir))
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if "outs" in file:
                     outs_list.append(os.path.join(root, file))
                if "trgs" in file:
                     trgs_list.append(os.path.join(root, file))

        if len(outs_list) == self.config["data_loader"]["args"]["num_folds"]:
            for i in range(len(outs_list)):
                outs = np.load(outs_list[i])
                trgs = np.load(trgs_list[i])
                all_outs.extend(outs)
                all_trgs.extend(trgs)

        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        if len(all_outs) == 0 or len(all_trgs) == 0:
            print("Warning: No valid data found in all_outs or all_trgs. Skipping metric calculation.")
            return

        r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_trgs, all_outs)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df["accuracy"] = accuracy_score(all_trgs, all_outs)
        df = df * 100
        file_name = self.config["name"] + "_classification_report.xlsx"
        report_save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_save_path)

        cm_file_name = self.config["name"] + "_confusion_matrix.torch"
        cm_save_path = os.path.join(save_dir, cm_file_name)
        torch.save(cm, cm_save_path)

        print(f"Metrics have been calculated and saved to {report_save_path} and {cm_save_path}.")
