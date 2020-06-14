# First steps
Before playing with the project:

    -> create and active virtualenv
    -> install mujoco-py (follow https://github.com/openai/mujoco-py)
    -> install required packages :
        - 'required.txt'      (tf-nightly)     [NVidia GPU]
        - 'required-rocm.txt' (tensorflow-rocm)[AMD GPU]

All scripts should be called from the project home directory 

# Dependancies
RadeoOpenCompute (Tensorflow): https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/develop-upstream/rocm_docs/tensorflow-install-basic.md
MuJoCo : https://github.com/openai/mujoco-py

# Status
All software has been written with tf-nightly 2.3.0. ROCm support for this distribution has not been provided yet.
As 2.2.0 differs from 2.3.0 in some crucial fragments, AMD-GPU version cannot be used yet. Set of the ROCm-dependancies
will be provided as soon as tensorflow-2.3.0 is provided.