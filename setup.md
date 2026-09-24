# conda install

conda create -n did python=3.12 -y
conda activate did

## 安装torch（>2.0）


pip install -r requirements.txt

cd smg_metric
pip install -e ".[dev,torch]"
