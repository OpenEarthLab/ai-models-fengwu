# ai-models-fengwu

`ai-models-fengwu` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run [Shanghai AI Laboratory's FengWu](https://github.com/OpenEarthLab/FengWu).

FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, arXiv preprint: 2304.02948, 2023.
<https://arxiv.org/abs/2304.02948>

FengWu was created by Kang Chen, Tao Han, Junchao Gong, Lei Bai, Fenghua Ling, Jing-Jia Luo, Xi Chen, Leiming Ma, Tianning Zhang, Rui Su, Yuanzheng Ci, Bin Li, Xiaokang Yang and Wanli Ouyang. It is released by Shanghai AI Laboratory.

The trained parameters of FengWu are made available under the terms of the BY-NC-SA 4.0 license.

The commercial use of these models is forbidden.

Please download the pre-trained Fengwu without transfer learning (fengwu_v1.onnx) from OneDrive drive: https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/EVA6V_Qkp6JHgXwAKxXIzDsBPIddo5RgDtGCBQ-sQbMmwg

Please download the pre-trained Fengwu with transfer learning (fengwu_v2.onnx): from OneDrive drive: https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/EZkFM7nQcEtBve6MsqlWaeIB_lmpa__hX0I8QYOPzf-X6A

See <https://github.com/OpenEarthLab/FengWu> for further details.

### Installation

To install the package, run:

```bash
pip install ai-models-fengwu
```

This will install the package and its dependencies, in particular the ONNX runtime. The installation script will attempt to guess which runtime to install. You can force a given runtime by specifying the the `ONNXRUNTIME` variable, e.g.:

```bash
ONNXRUNTIME=onnxruntime-gpu pip install ai-models-fengwu
```