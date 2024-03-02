# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import numpy as np
import onnxruntime as ort
from ai_models.model import Model

LOG = logging.getLogger(__name__)


class FengWu(Model):
    # Download
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/fengwu/{file}"
    )
    download_files = ["fengwu.onnx", "data_mean.npy", "data_std.npy"]

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = ['10u', '10v', '2t', 'msl']
    param_level_pl = (
        ['z', 'q', 'u', 'v', 't'],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    # Output
    expver = "fenw"

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads
        self.hour_steps = 6
        self.lagged = [-6, 0]

    def get_meanvar(self):
        path = os.path.join(self.assets, self.download_files[1])
        info = ("Loading %s from %s" % (self.download_files[1], path))
        LOG.info(info)
        self.data_mean = np.load(path)
        self.data_mean = self.data_mean[:, np.newaxis, np.newaxis]

        path = os.path.join(self.assets, self.download_files[2])
        info = ("Loading %s from %s" % (self.download_files[2], path))
        LOG.info(info)
        self.data_std = np.load(path)
        self.data_std = self.data_std[:, np.newaxis, np.newaxis]

    def get_input(self):
        fields_pl = self.fields_pl

        param, level = self.param_level_pl
        fields_pl = fields_pl.sel(param=param, level=level)
        fields_pl = fields_pl.order_by('valid_datetime', param=param, level=level)

        fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
        fields_pl_numpy1, fields_pl_numpy2 = np.split(fields_pl_numpy, 2, axis=0)

        fields_sfc = self.fields_sfc
        fields_sfc = fields_sfc.sel(param=self.param_sfc)
        fields_sfc = fields_sfc.order_by('valid_datetime', param=self.param_sfc)

        fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)
        fields_sfc_numpy1, fields_sfc_numpy2 = np.split(fields_sfc_numpy, 2, axis=0)

        input1 = np.concatenate([fields_sfc_numpy1, fields_pl_numpy1], axis=0)
        input2 = np.concatenate([fields_sfc_numpy2, fields_pl_numpy2], axis=0)

        input1_after_norm = (input1 - self.data_mean) / self.data_std
        input2_after_norm = (input2 - self.data_mean) / self.data_std

        input = np.concatenate((input1_after_norm, input2_after_norm), axis=0)[np.newaxis, :, :, :]
        input = input.astype(np.float32)

        self.template_pl = fields_pl[: len(fields_pl) // len(self.lagged)]
        self.template_sfc = fields_sfc[: len(fields_sfc) // len(self.lagged)]  
        return input

    def run(self):
        self.get_meanvar()
        input = self.get_input()

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena=False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        # Increase the number for faster inference and more memory consumption
        options.intra_op_num_threads = self.num_threads

        model_6 = os.path.join(self.assets, self.download_files[0])

        with self.timer(f"Loading {model_6}"):
            ort_session_6 = ort.InferenceSession(
                model_6,
                sess_options=options,
                providers=self.providers,
            )

        with self.stepper(6) as stepper:
            for i in range(self.lead_time // self.hour_steps):
                step = (i + 1) * self.hour_steps

                output = ort_session_6.run(None, {'input':input})[0]
                input = np.concatenate((input[:, 69:], output[:, :69]), axis=1)

                # Save the results
                output = (output[0, :69] * self.data_std) + self.data_mean
                sfc_data = output[: len(self.param_sfc)]
                pl_data = output[len(self.param_sfc):]

                for data, f in zip(sfc_data, self.template_sfc):
                    self.write(data, template=f, step=step)

                for data, f in zip(pl_data, self.template_pl):
                    self.write(data, template=f, step=step)

                stepper(i, step)