import logging

import torch
import torch.nn.functional as F

from kraken.models.ctc import CTCRecognitionInferenceMixin
from kraken.containers import BaselineOCRRecord


logger = logging.getLogger(__name__)


def _patched_recognize_baseline_lines(self, lines, segmentation):
    max_len = max([seq.shape[2] for seq, *_ in lines])

    seqs = torch.stack([
        F.pad(seq, pad=(0, max_len - seq.shape[2]))
        for seq, *_ in lines
    ])

    # Kraken 7.1 bug fix:
    # seq_lens was created on CPU even when the model/input was on CUDA.
    seq_lens = torch.LongTensor(
        [seq.shape[2] for seq, *_ in lines],
        device=seqs.device
    )

    preds, olens = self._rec_predict(seqs, seq_lens)

    for idx, (pred, olen) in enumerate(zip(preds, olens)):
        self.net_scale = lines[idx][0].shape[2] / olen.item()

        self.in_scale = lines[idx][1].width / (
            lines[idx][0].shape[2] - 2 * self._inf_config.padding
        )

        pred_str = ''.join(x[0] for x in pred)

        pos = []
        conf = []

        for _, start, end, c in pred:
            pos.append([
                self._scale_val(start, 0, lines[idx][1].width),
                self._scale_val(end, 0, lines[idx][1].width)
            ])
            conf.append(c)

        rec = BaselineOCRRecord(
            pred_str,
            pos,
            conf,
            segmentation.lines[lines[idx][2]],
            logits=(
                self.outputs[idx, ..., :olen].clone()
                if self._inf_config.return_logits
                else None
            ),
            image=(
                lines[idx][1]
                if self._inf_config.return_line_image
                else None
            )
        )

        if self._inf_config.bidi_reordering:
            logger.debug('BiDi reordering record.')
            yield (
                rec.logical_order(
                    base_dir=(
                        self._inf_config.bidi_reordering
                        if self._inf_config.bidi_reordering in ('L', 'R')
                        else None
                    )
                ),
                lines[idx][2]
            )
        else:
            logger.debug('Emitting raw record')
            yield rec.display_order(None), lines[idx][2]


def apply_kraken_monkey_patch():
    """Apply compatibility fix for Kraken 7.1 CUDA inference."""
    CTCRecognitionInferenceMixin._recognize_baseline_lines = (
        _patched_recognize_baseline_lines
    )