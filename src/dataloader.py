import pandas as pd
import numpy as np
import torch
import logging
import os
import os.path as osp
import itertools

import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor

import datatable as dt
from datetime import datetime
from datatable import f, join, sort
from src.util import find_parallel_edges

def format_dataset(inPath):
    r"""
    Turn text attributed dataset into a dataset only contains numbers.
    """
    outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

    raw = dt.fread(inPath, columns=dt.str32)

    currency = dict()
    paymentFormat = dict()
    bankAcc = dict()
    account = dict()

    def get_dict_val(name, collection):
        if name in collection:
            val = collection[name]
        else:
            val = len(collection)
            collection[name] = val
        return val

    header = "EdgeID,from_id,to_id,Timestamp,\
    Amount Sent,Sent Currency,Amount Received,Received Currency,\
    Payment Format,Is Laundering\n"

    firstTs = -1

    with open(outPath, "w") as writer:
        writer.write(header)
        for i in range(raw.nrows):
            datetime_object = datetime.strptime(raw[i, "Timestamp"], "%Y/%m/%d %H:%M")
            ts = datetime_object.timestamp()
            day = datetime_object.day
            month = datetime_object.month
            year = datetime_object.year
            hour = datetime_object.hour
            minute = datetime_object.minute

            if firstTs == -1:
                startTime = datetime(year, month, day)
                firstTs = startTime.timestamp() - 10

            ts = ts - firstTs

            cur1 = get_dict_val(raw[i, "Receiving Currency"], currency)
            cur2 = get_dict_val(raw[i, "Payment Currency"], currency)

            fmt = get_dict_val(raw[i, "Payment Format"], paymentFormat)

            fromAccIdStr = raw[i, "From Bank"] + raw[i, 2]
            fromId = get_dict_val(fromAccIdStr, account)

            toAccIdStr = raw[i, "To Bank"] + raw[i, 4]
            toId = get_dict_val(toAccIdStr, account)

            amountReceivedOrig = float(raw[i, "Amount Received"])
            amountPaidOrig = float(raw[i, "Amount Paid"])

            isl = int(raw[i, "Is Laundering"])

            line = "%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n" % (
                i,
                fromId,
                toId,
                ts,
                amountPaidOrig,
                cur2,
                amountReceivedOrig,
                cur1,
                fmt,
                isl,
            )

            writer.write(line)

    formatted = dt.fread(outPath)
    formatted = formatted[:, :, sort(3)]

    formatted.to_csv(outPath)


class AMLData:
    csv_names = {
        "Small_LI": "LI-Small_Trans.csv",
        "Small_HI": "HI-Small_Trans.csv",
        "Medium_LI": "LI-Medium_Trans.csv",
        "Medium_HI": "HI-Medium_Trans.csv",
        "Large_LI": "LI-Large_Trans.csv",
        "Large_HI": "HI-Large_Trans.csv",
    }

    def __init__(self, config):
        self.config = config
        self.root_dir = osp.join(config.data_path, config.data)

    def get_data(self):
        if osp.exists(osp.join(self.root_dir, "data.pt")):
            cached_data = torch.load(osp.join(self.root_dir, "data.pt"))
            # Unpack the dictionary into a tuple
            return (
                cached_data["tr_data"],
                cached_data["val_data"],
                cached_data["te_data"],
                cached_data["tr_inds"],
                cached_data["val_inds"],
                cached_data["te_inds"],
            )
        else:
            return self.process(self.config)

    def process(self, config):
        """Loads the AML transaction data.

        1. The data is loaded from the csv and the necessary features are chosen.
        2. The data is split into training, validation and test data.
        3. PyG Data objects are created with the respective data splits.
        """

        format_dataset(osp.join(self.root_dir, self.csv_names[config.data]))

        transaction_file = osp.join(
            self.root_dir, "formatted_transactions.csv"
        )  # replace this with your path to the respective AML data objects
        df_edges = pd.read_csv(transaction_file)
        df_edges = df_edges.sort_values(by="Timestamp")

        logging.info(f"Available Edge Features: {df_edges.columns.tolist()}")

        df_edges["Timestamp"] = df_edges["Timestamp"] - df_edges["Timestamp"].min()

        max_n_id = df_edges.loc[:, ["from_id", "to_id"]].to_numpy().max() + 1
        df_nodes = pd.DataFrame(
            {"NodeID": np.arange(max_n_id), "Feature": np.ones(max_n_id)}
        )
        timestamps = torch.Tensor(df_edges["Timestamp"].to_numpy())
        y = torch.LongTensor(df_edges["Is Laundering"].to_numpy())

        logging.info(
            f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%"
        )
        logging.info(
            f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}"
        )
        logging.info(f"Number of transactions = {df_edges.shape[0]}")

        edge_features = [
            "Timestamp",
            "Amount Received",
            "Received Currency",
            "Payment Format",
        ]
        node_features = ["Feature"]

        logging.info(f"Edge features being used: {edge_features}")
        logging.info(
            f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)'
        )

        x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
        edge_index = torch.LongTensor(
            df_edges.loc[:, ["from_id", "to_id"]].to_numpy().T
        )
        edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

        simp_edge_batch = find_parallel_edges(edge_index)

        n_days = int(timestamps.max() / (3600 * 24) + 1)
        n_samples = y.shape[0]
        logging.info(
            f"number of days and transactions in the data: {n_days} days, {n_samples} transactions"
        )

        # data splitting
        daily_irs, weighted_daily_irs, daily_inds, daily_trans = (
            [],
            [],
            [],
            [],
        )  # irs = illicit ratios, inds = indices, trans = transactions
        for day in range(n_days):
            l = day * 24 * 3600
            r = (day + 1) * 24 * 3600
            day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
            daily_irs.append(y[day_inds].float().mean())
            weighted_daily_irs.append(
                y[day_inds].float().mean() * day_inds.shape[0] / n_samples
            )
            daily_inds.append(day_inds)
            daily_trans.append(day_inds.shape[0])

        split_per = [0.6, 0.2, 0.2]
        daily_totals = np.array(daily_trans)
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i, j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v / split_totals_sum for v in split_totals]
                split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
                score = max(split_error)  # - (split_totals_sum/total) + 1
                split_scores[(i, j)] = score
            else:
                continue

        i, j = min(split_scores, key=split_scores.get)
        # split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
        logging.info(f"Calculate split: {split}")

        # Now, we seperate the transactions based on their indices in the timestamp array
        split_inds = {k: [] for k in range(3)}
        for i in range(3):
            for day in split[i]:
                split_inds[i].append(
                    daily_inds[day]
                )  # split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

        tr_inds = torch.cat(split_inds[0])
        val_inds = torch.cat(split_inds[1])
        te_inds = torch.cat(split_inds[2])
        

        logging.info(
            f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}"
        )
        logging.info(
            f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}"
        )
        logging.info(
            f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}"
        )

        # Creating the final data objects
        tr_x, val_x, te_x = x, x, x
        e_tr = tr_inds.numpy()
        e_val = np.concatenate([tr_inds, val_inds])

        tr_edge_index, tr_edge_attr, tr_y, tr_edge_times = (
            edge_index[:, e_tr],
            edge_attr[e_tr],
            y[e_tr],
            timestamps[e_tr],
        )
        val_edge_index, val_edge_attr, val_y, val_edge_times = (
            edge_index[:, e_val],
            edge_attr[e_val],
            y[e_val],
            timestamps[e_val],
        )
        te_edge_index, te_edge_attr, te_y, te_edge_times = (
            edge_index,
            edge_attr,
            y,
            timestamps,
        )

        tr_simp_edge_batch, val_simp_edge_batch, te_simp_edge_batch = simp_edge_batch[e_tr], simp_edge_batch[e_val], simp_edge_batch



        tr_data = GraphData(
            x=tr_x,
            y=tr_y,
            edge_index=tr_edge_index,
            edge_attr=tr_edge_attr,
            timestamps=tr_edge_times,
            simp_edge_batch=tr_simp_edge_batch,
        )
        val_data = GraphData(
            x=val_x,
            y=val_y,
            edge_index=val_edge_index,
            edge_attr=val_edge_attr,
            timestamps=val_edge_times,
            simp_edge_batch=val_simp_edge_batch,
        )
        te_data = GraphData(
            x=te_x,
            y=te_y,
            edge_index=te_edge_index,
            edge_attr=te_edge_attr,
            timestamps=te_edge_times,
            simp_edge_batch=te_simp_edge_batch,
        )

        # Normalize data
        tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = (
            z_norm(tr_data.edge_attr),
            z_norm(val_data.edge_attr),
            z_norm(te_data.edge_attr),
        )

        logging.info(f"train data object: {tr_data}")
        logging.info(f"validation data object: {val_data}")
        logging.info(f"test data object: {te_data}")

        torch.save(
            {
                "tr_data": tr_data,
                "val_data": val_data,
                "te_data": te_data,
                "tr_inds": tr_inds,
                "val_inds": val_inds,
                "te_inds": te_inds,
            },
            osp.join(self.root_dir, "data.pt"),
        )

        logging.info(f'Data is saved to: {osp.join(self.root_dir, "data.pt")}')

        return tr_data, val_data, te_data, tr_inds, val_inds, te_inds


class GraphData(Data):
    """This is the homogenous graph object we use for GNN training if reverse MP is not enabled"""

    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        readout: str = "edge",
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = "ce"
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps
        elif edge_attr is not None:
            self.timestamps = edge_attr[:, 0].clone()
        else:
            self.timestamps = None


def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std
