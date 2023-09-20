import torch

def run():
    print(torch.__version__)

    torch.cuda.is_available()

    if torch.cuda.is_available():
        print("CUDA GPU Compatibility Available")
    else:
        print("CUDA GPU Compatibility Not Available")


if __name__ == '__main__':
    run()

'''
if torch version ended with +cpu, therefore no GPU usage in overall processes
hence, install CUDA Toolkit and make it ended with +cu1xx where x stand for versions

install torch with gpu capabilities
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 [Windows on Sept 2023]
'''
