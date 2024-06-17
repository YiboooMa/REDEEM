# REDEEM

## Mitigating Energy Consumption in Heterogeneous Mobile Networks through Data-Driven Optimization
by Yibo Ma, Tong Li, Yan Zhou, Li Yu, Depeng Jin

### Abstract
5G networks, with their notable energy consumption, pose a significant challenge. Traditional energy-saving methods, effective for 4G, struggle in heterogeneous 4G and 5G networks. In this paper, we propose the pRoactivE Data-drivEn Energy Saving Method (REDEEM) to mitigate energy consumption in heterogeneous 4G and 5G mobile networks. REDEEM spatially divides the network into meshes based on cell overlaps, predicts cell traffic for proactive control, and selects active cells within each mesh. Our framework includes energy efficiency profiling for each mesh and offloads 5G traffic onto overlapping 4G cells to reduce 5G energy usage. Experiments based on the Nanchang mobile networks validate REDEEM's effectiveness, yielding energy savings of 3442.72 MWh over a week. Notably, our approach achieves a 53.10\% energy-saving rate, surpassing threshold-based methods by 38.85\%, optimization-based methods by 18.15\%, and fluid capacity engine by 14.79\%. It minimally impacts service quality, with less than four parts per million traffic missed. Experimental results also demonstrate REDEEM's robustness across various temporal, spatial, and traffic load scenarios.

### Replication data and code
* Power_RRU_Model.py
   * Python code for the *RRU Energy Consumption Model* in Section III.B of the paper
* Mesh_Division.py
   * Python code to replicate the *Mesh Division of Mobile Networks* in Section V.B of the paper
* Traffic_Prediction_Model/prediction.py
   * Python code to replicate the *Traffic Prediction for Cells* in Section V.C of the paper
* Traffic_Transfer.py
   * Python code to replicate the *Active Cells Selection* in Section V.D of the paper
* Power_Element.py
   * Python code to replicate the *Energy Efficiency Profiling* in Section V.D of the paper
* Traffic5G_Offloading.py
   * Python code to replicate the *Traffic Offloading of 5G Cells* in Section V.E of the paper
* data
   * Partial Mobile network data of NanChang city