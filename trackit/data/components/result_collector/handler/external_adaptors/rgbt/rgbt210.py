# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for RGBT210 evaluation

import os
import io
import zipfile
from typing import Sequence
import numpy as np
from rgbt import RGBT210
from tabulate import tabulate

from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.data.protocol.eval_output import SequenceEvaluationResult_SOT
from .utils import serial_process, iou, cle, SequenceMetric, DatasetMetrics

from ....progress_tracer import EvaluationProgress
from ... import EvaluationResultHandler


class RGBT210EvaluationMetricCalculator:
    def __init__(self, pr_thr=np.linspace(0, 50, 51), pr_val_idx=20, sr_thr=np.linspace(0, 1, 21)):
        self._dataset = RGBT210()
        self._pr_thr = pr_thr
        self._pr_val_idx = pr_val_idx
        self._sr_thr = sr_thr

    def __call__(self, seq_name: str, predicted_bboxes: np.ndarray) -> SequenceMetric:
        gt = self._dataset[seq_name]['visible']

        processed_bboxes = [r for r in predicted_bboxes]
        processed_bboxes[0] = gt[0]

        pr_res = serial_process(cle, processed_bboxes, gt)
        sr_res = serial_process(iou, processed_bboxes, gt)

        pr_cell = []
        for i in self._pr_thr:
            pr_cell.append(np.sum(pr_res <= i) / len(pr_res))
        pr_val = pr_cell[self._pr_val_idx]

        sr_cell = []
        for i in self._sr_thr:
            sr_cell.append(np.sum(sr_res > i) / len(sr_res))
        sr_val = np.mean(sr_cell)

        metric = SequenceMetric(
            sequence_name=seq_name,
            precision_score=pr_val,
            success_score=sr_val,
        )

        return metric


class RGBT210EvaluationToolFileWriter:
    def __init__(self, output_folder: str, output_file_name: str):
        self._output_file_path_prefix = os.path.join(output_folder, output_file_name)
        self._zipfile = zipfile.ZipFile(self._output_file_path_prefix + '.zip', 'w', zipfile.ZIP_DEFLATED)
        self._duplication_check = set()

    def write(self, tracker_name: str, sequence_name: str,
              predicted_bboxes: np.ndarray, time_costs: np.ndarray):
        assert (tracker_name, sequence_name) not in self._duplication_check, "duplicated sequence name detected"
        self._duplication_check.add((tracker_name, sequence_name))

        with io.BytesIO() as result_file_content:
            np.savetxt(result_file_content, predicted_bboxes, fmt='%.4f',
                       delimiter=',')
            self._zipfile.writestr(
                f'{tracker_name}/{sequence_name}.txt',
                result_file_content.getvalue()
            )

        with io.BytesIO() as time_file_content:
            np.savetxt(time_file_content, time_costs, fmt='%.8f')
            self._zipfile.writestr(
                f'{tracker_name}/{sequence_name}_time.txt',
                time_file_content.getvalue()
            )

    def close(self):
        self._zipfile.close()


class RGBT210EvaluationToolAdaptor(EvaluationResultHandler):
    def __init__(self, tracker_name: str, output_folder: str, file_name: str = 'RGBT210', rasterize_bbox: bool = True):
        self.output_folder = output_folder

        if output_folder is not None:
            self._writer = RGBT210EvaluationToolFileWriter(output_folder, file_name)

        self._tracker_name = tracker_name
        self._rasterize_bbox = rasterize_bbox
        self._metric_calculator = RGBT210EvaluationMetricCalculator()
        self._dataset_metrics = DatasetMetrics()

    def accept(self, evaluation_results: Sequence[SequenceEvaluationResult_SOT],
               evaluation_progresses: Sequence[EvaluationProgress]):
        for evaluation_result, evaluation_progress in zip(evaluation_results, evaluation_progresses):
            predicted_bboxes = evaluation_result.output_box
            if self._rasterize_bbox:
                predicted_bboxes = bbox_rasterize(predicted_bboxes)
            predicted_bboxes = bbox_xyxy_to_xywh(predicted_bboxes)

            metric = self._metric_calculator(evaluation_result.sequence_info.sequence_name, predicted_bboxes)
            self._dataset_metrics.add_metric(metric)

            print(
                f'{evaluation_result.sequence_info.sequence_name}: success {metric.success_score: .04f}, prec {metric.precision_score: .04f}')

            if self.output_folder is not None:
                self._writer.write(
                    self._tracker_name,
                    evaluation_result.sequence_info.sequence_name,
                    predicted_bboxes,
                    evaluation_result.time_cost
                )

    def close(self):
        if self.output_folder is not None:
            self._writer.close()

        print(f'RGBT210 summary metrics: \n' + tabulate(self._dataset_metrics.compute().items(),
                                                        headers=('metric', 'value'),
                                                        floatfmt=".4f"), flush=True)
