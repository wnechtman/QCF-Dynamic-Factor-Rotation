import pandas as pd
import statsmodels as sm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from support import fama_french, set_first
from fredapi import Fred
fred = Fred(api_key="11ae3d2ef9363b6c65ce74df68755758")
plt.style.use("ggplot")

# some indexes make it tricky, might just want to get MoM change and then plot cumulative

def csi(since_y="1978"):
    # Get UM Consumer Sentiment as cumulative return from since_y
    # Continuous since 1978, 1966 Q1 = 100, need to reset to 1978-01-01
    df = pd.DataFrame(fred.get_series("UMCSENT", observation_start=since_y), columns=["UMCSENT"])
    df = df.resample("M").last().to_period("M")
    df.index = pd.to_datetime(df.index.to_timestamp())
    return ((df.pct_change() + 1).cumprod() - 1).dropna() + 1

def cnfci():
    # Get Chicago Fed National Financial Conditions Index
    df = pd.DataFrame(fred.get_series("NFCI"), columns=["NFCI"])
    df = df.resample("M").last().to_period("M")
    df.index = pd.to_datetime(df.index.to_timestamp())
    return df

def slfsi():
    # Get St. Louis Fed Financial Stress Index
    df = pd.DataFrame(fred.get_series("STLFSI3"), columns=["STLFSI3"])
    df = df.resample("M").last().to_period("M")
    df.index = pd.to_datetime(df.index.to_timestamp())
    return df

def cpi(since_y="1971"):
    # Get Consumer Price Index as cumulative return from since_y
    df = pd.DataFrame(fred.get_series("CPIAUCSL", observation_start=since_y), columns=["CPIAUCSL"])
    df = df.resample("M").last().to_period("M")
    df.index = pd.to_datetime(df.index.to_timestamp())
    return ((df.pct_change() + 1).cumprod() - 1).dropna() + 1

def hys():
    # High yield spread
    df = pd.DataFrame(fred.get_series("BAMLH0A0HYM2"), columns=["BAMLH0A0HYM2"])
    df = df.resample("M").last().to_period("M")
    df.index = pd.to_datetime(df.index.to_timestamp())
    return df



data = pd.concat([cnfci(), csi(), slfsi(), cpi(), hys()], axis=1)
data.dropna(inplace=True)

# Use MoM change
data_mom = data.pct_change().dropna()
data_mom.head()

data_mom = data.pct_change().dropna()

print("Reducing Data...\n")
pca = PCA(n_components=2)
red_mom = pca.fit_transform(data_mom)

print("Clustering...\n")
kmeans = KMeans(n_clusters=4)
kmeans.fit(red_mom)

print("Plotting...\n")
# Mesh step size, lower for better quality
h = 0.05

# Buffer for min/max
buff = 0.1

x_min, x_max = red_mom[:, 0].min() - buff, red_mom[:, 0].max() + buff
y_min, y_max = red_mom[:, 1].min() - buff, red_mom[:, 1].max() + buff
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,8))
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)
plt.axhline(y=0, color="black", linewidth=1, linestyle="--")
plt.axvline(x=0, color="black", linewidth=1, linestyle="--")
plt.plot(red_mom[:, 0], red_mom[:, 1], "k.", markersize=5)
plt.xlabel(f"PC1({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2({pca.explained_variance_ratio_[1]*100:.2f}%)")
total_var = pca.explained_variance_ratio_.sum()*100
t = r"PCA $n=2$ Reduction of Indicators" + f"\nExplained Variance: {total_var:.2f}%"
plt.title(t)
plt.show()