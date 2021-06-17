### Python Evaluation Tool for MVDet

This is simply the python translation of a MATLAB　Evaluation tool used to evaluate detection result created by P. Dollar.
Translated by [Zicheng Duan](https://github.com/ZichengDuan)

#### Purpose
   Allowing the project to run purely in Python without using MATLAB Engine.

#### Critical information before usage
   1. This API is only tested and deployed in this project: [hou-yz/MVDet](https://github.com/hou-yz/MVDet), might not be compatible with other projects.
   2. The detection result using this API **is a little bit lower** (approximately 0~2% decrease in MODA, MODP) than that using official MATLAB evaluation tool, the reason might be that the Hungarian Algorithm implemented in sklearn is a little bit different with the one implemented by P. Dollar, hence resulting in different results.   
   Therefore, **please use the official MATLAB API if you want to obtain the same evaluation result shown in the paper**. This Python API is only used for convenience.
   3. The training process would not be affected by this API.
