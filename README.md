# OFA
## Robust Anomaly Detection in Feature-evolving Time Series [PDF](https://doi.org/10.1093/comjnl/bxaa174 "Downdoald the paper from here")

## Abstract
_This paper addresses the anomaly detection problem in feature-evolving systems
such as server machines, cyber security, financial markets and so forth where
in every millisecond, N -dimensional feature-evolving heterogeneous time series
are generated.
However, due to stochasticity and uncertainty in evolving
heterogeneous time series coupled with temporal dependencies, their anomaly
detection are extremely challenging. Furthermore, it is practically impossible to
train an anomaly detection model per single time series across millions of metrics,
leave alone memory space required to maintain the model and evolving data
points in memory for timely processing in feature-evolving data streams. Thus,
this paper proposes One sketch F its all Algorithm (OFA), which is a real-time
stochastic recurrent deep neural network anomaly detector built on assumption-
free probabilistic conditional Quantile Regression (QR) with well-calibrated predictive
uncertainty estimates. The proposed framework is capable of detecting anomalies
robustly, accurately and efficiently in real-time while handling randomness and
variabilities in feature-evolving heterogeneous time series. Extensive experiments
and rigorous evaluation on large-scale real world data sets showcases that
OFA outperforms other competitive state-of-the-art anomaly detector methods._

## ofa.py 
This file contains OFA implemetation algorithm 

## Datasets
- Art.tar.gz is an artificial dataset
* edf_stocks.csv [can be downloaded here](https://github.com/Amossys-team/SPOT "edf stock market dataset")
- hrrs -High Rack Storage System [available from here](https://www.kaggle.com/inIT-OWL/high-storage-system-data-for-energy-optimization/data "hrss dataset link")
* NYC Taxi [can be downloaded here](https://data.cityofnewyork.us/Transportation/2014-Yellow-Taxi-Trip-Data/gkne-dk5s "NYC Taxi Dataset")
- credit [can be downloaded here](https://www.kaggle.com/mlg-ulb/creditcardfraud "Credit card transaction dataset")

## Dependencies
1. Tensoflow 2
2. Keras 2.3.1
3. Python 3.6
4. Matplotlib
5. tqdm

If you find this code useful in your research, please, consider citing our paper:

# Reference
```
@ARTICLE{wamburacmpjrn2020,
	AUTHOR = "Stephen Wambura and Jianbin Huang and He Li",
	TITLE = "Robust Anomaly Detection in Time series",
	JOURNAL = {Oxford computer journal},
	VOLUME = {},
	NUMBER = {},
	YEAR = {2020},}
```
# License
OFA is distributed under Apache 2.0 license.

Contact: Stephen Wambura (ceo@stephenwambura.com)
