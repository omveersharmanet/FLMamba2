# FLMamba2

## FLMamba2: A Frequency-Aware Lightweight State Space Network for Multivariate Time Series Forecasting



## Contributions :trophy:

- We propose FLMamba2, A Frequency-Aware Lightweight State Space Network for Multivariate Time Series Forecasting. 
- We evaluate the performance of FLMamba2, which not only has low GPU memory required and short time for forecasts but also maintains superior performance compared to the representative and state-of-the-art models. 
- We conduct extensive experiments to further delve deeper into Mamba's potential in time series forecasting tasks.

## 🌟 Getting Start

### 🛠️ Installation

```bash
pip install -r requirements.txt
```

### 📦 Datasets



### 🚀 Train and evaluate

```bash
# ECL
bash ./scripts/multivariate_forecasting/ECL/FS_Mamba2.sh
# Exchange
bash ./scripts/multivariate_forecasting/Exchange/FS_Mamba2.sh
# Traffic
bash ./scripts/multivariate_forecasting/Traffic/FS_Mamba2.sh
# Weather
bash ./scripts/multivariate_forecasting/Weather/FS_Mamba2.sh
# Solar-Energy
bash ./scripts/multivariate_forecasting/SolarEnergy/FS_Mamba2.sh
# PEMS
bash ./scripts/multivariate_forecasting/PEMS/FS_Mamba2_03.sh
bash ./scripts/multivariate_forecasting/PEMS/FS_Mamba2_04.sh
bash ./scripts/multivariate_forecasting/PEMS/FS_Mamba2_07.sh
bash ./scripts/multivariate_forecasting/PEMS/FS_Mamba2_08.sh
# ETT
bash ./scripts/multivariate_forecasting/ETT/FS_Mamba2_ETTm1.sh
bash ./scripts/multivariate_forecasting/ETT/FS_Mamba2_ETTm2.sh
bash ./scripts/multivariate_forecasting/ETT/FS_Mamba2_ETTh1.sh
bash ./scripts/multivariate_forecasting/ETT/FS_Mamba2_ETTh2.sh
```


## :pray: Acknowledgement 

We are grateful for the following awesome projects when implementing FS_Mamba2 or FL_Mamba2:

- [iTransformer](https://github.com/thuml/iTransformer)
- [Mamba](https://github.com/state-spaces/mamba)


## 🤝 Join the Collaboration
We warmly welcome your participation! Whether you have ideas for improvements, feature additions, or bug fixes, feel free to open an issue or submit a pull request.


