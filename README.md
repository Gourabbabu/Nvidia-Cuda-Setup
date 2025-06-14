# Setup-NVIDIA-GPU-for-Deep-Learning

## Step 1: NVIDIA Video Driver
### Steps:
- Visit the link below.
- Select your GPU model (you can find it using dxdiag or Device Manager).
- Download and install the latest driver.
- Restart your PC after installation.

You should install the latest version of your GPUs driver. You can download drivers here:
 - [NVIDIA GPU Drive Download](https://www.nvidia.com/Download/index.aspx)

## Step 2: Visual Studio C++
### Steps:
- Download and install the Community Edition (free).
- During installation, select:
- Desktop development with C++
- Optional: Python Development if you're using VS for coding too.
- Complete the setup and restart if prompted.

You will need Visual Studio, with C++ installed. By default, C++ is not installed with Visual Studio, so make sure you select all of the C++ options.
 - [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)

## Step 3: Anaconda/Miniconda
### Steps:
- Install Anaconda for Windows (Python 3.x version).
- Open the Anaconda Prompt.
- Create a new environment:
```cmd
conda create -n dl_env python=3.10
conda activate dl_env
```
You will need anaconda to install all deep learning packages
 - [Download Anaconda](https://www.anaconda.com/download/success)

## Step 4: CUDA Toolkit
### Steps:
- Choose a version compatible with your PyTorch install (e.g., CUDA 11.8).
- Install it using the local installer (easier and more stable).
- During installation, select:
- CUDA Toolkit
- Drivers (optional if already installed)
- Visual Studio Integration (optional)
 - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

## Step 5: cuDNN
### Steps:
- Register/log in to the NVIDIA Developer Program.
- Choose the cuDNN version that matches your CUDA version.
- Download the ZIP file for Windows.
- Extract and copy the contents to your CUDA folder:
```
bin → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin

lib → ...\lib\x64

include → ...\include
```

 - [Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)


## Step 6: Install PyTorch 
### Steps:
- Use the selector on the site to choose:
- Package: Conda
- Language: Python
- Compute Platform: CUDA 11.8 (or whatever you installed)

Example installation command:
```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
 - [Install PyTorch](https://pytorch.org/get-started/locally/)




## Finally run the following script to test your GPU

```python
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```
