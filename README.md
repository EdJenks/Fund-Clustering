# Fund Clustering

This project leverages time series clustering techniques with the goal of clustering similar funds. The idea being that similar investment funds tend to exhibit similar price action.

Data.csv contains historical data of the funds included. Each fund has a maximum of three time points and data about its composition at this point. This dataset might contain inconsistencies and data quality issues.

cluster_composition is a CSV file that contains information about the cluster composition that's produced by the clustering algorithm.

Fund Clustering is a PDF that explains the feature engineering and agglomerative clustering algorithm in detail. In addition, some potential improvements are included in this pdf.

To run the app, users need to install the requirements, and simply call qd.py.
