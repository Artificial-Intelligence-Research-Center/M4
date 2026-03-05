### 🔧Install environment

1. Create environment with conda:

```
conda create -n retfound python=3.11.0 -y
conda activate retfound
```

2. Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install ipykernel
pip install termcolor
python -m ipykernel install --user --name retfound --display-name "Python (retfound)"
git clone https://github.com/Artificial-Intelligence-Research-Center/M4.git
cd M4
pip install -r requirements.txt
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
```
