from abc import ABC, abstractmethod
from functools import partial
from typing import Optional

import pandas as pd
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.wrappers import MultioutputWrapper


class NetworkSharedMultioutputWrapper(MultioutputWrapper):
    """A MultioutputWrapper that shares the network between the metrics.

    Adapted from torchmetrics.wrappers.FeatureShare, but this one only shares the network.

    Args:
        base_metric: Metric being wrapped.
        num_outputs: Expected dimensionality of the output dimension.
            This parameter is used to determine the number of distinct metrics we need to track.
        output_dim:
            Dimension on which output is expected. Note that while this provides some flexibility, the output dimension
            must be the same for all inputs to update. This applies even for metrics such as `Accuracy` where the labels
            can have a different number of dimensions than the predictions. This can be worked around if the output
            dimension can be set to -1 for both, even if -1 corresponds to different dimensions in different inputs.
        remove_nans:
            Whether to remove the intersection of rows containing NaNs from the values passed through to each underlying
            metric. Proper operation requires all tensors passed to update to have dimension ``(N, ...)`` where N
            represents the length of the batch or dataset being passed in.
        squeeze_outputs:
            If ``True``, will squeeze the 1-item dimensions left after ``index_select`` is applied.
            This is sometimes unnecessary but harmless for metrics such as `R2Score` but useful
            for certain classification metrics that can't handle additional 1-item dimensions.
    """

    def __init__(
        self,
        base_metric: Metric,
        num_outputs: int,
        output_dim: int = -1,
        remove_nans: bool = True,
        squeeze_outputs: bool = True,
    ) -> None:
        super().__init__(
            base_metric, num_outputs, output_dim, remove_nans, squeeze_outputs
        )
        assert isinstance(self.metrics, nn.ModuleList)
        first_net = self.metrics[0]
        if hasattr(first_net, "feature_network"):
            network_to_share = getattr(first_net, first_net.feature_network)
            for metric in self.metrics[1:]:
                setattr(metric, metric.feature_network, network_to_share)

class TaskMetric(nn.Module, ABC):
    def __init__(self, eval_types: list[str]):
        super().__init__()
        # metric
        self.eval_types = eval_types
        self.metric_wrapper = partial(
            NetworkSharedMultioutputWrapper,
            num_outputs=len(self.eval_types),
            remove_nans=False,
        )

    @abstractmethod
    def update_metrics(self):
        raise NotImplementedError

    def reset_metrics(self, **kwargs):
        """Reset all underlying metrics by calling their reset method."""
        for metric in self.metrics.values():
            metric.reset()

    def compute_metrics(self, prefix: Optional[str] = None) -> dict[str, float]:
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        # compute
        metrics: dict[str, list[Tensor]] = {}
        for key, metric in self.metrics.items():
            results = metric.compute()
            if isinstance(results, dict):
                # results from MetricCollection is a dict
                metrics.update(results)
            else:
                metrics[key] = results
        # aggregate
        outputs = {
            f"{prefix}{eval_type}/{key}": float(f"{result.item():.4f}")
            for key, values in metrics.items()
            for eval_type, result in zip(self.eval_types, values)
        }
        return outputs

    @staticmethod
    def print_metrics(data):
        # Split the keys into two parts and use them as multi-index
        index = pd.MultiIndex.from_tuples([key.split("/") for key in data.keys()])
        # Create the DataFrame
        df = pd.DataFrame(list(data.values()), index=index)
        # Unstack the DataFrame to get the desired format
        df = df.unstack(level=0)
        df.columns = df.columns.droplevel(0)
        print(df.T)
