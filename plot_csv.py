#%%
import matplotlib.pyplot as plt
import pandas as pd
#%%
#PUT THE INFO YOU WANT
species="O3"
name_station="1461A"
units='ppb'
out_dir = "/home/agkiokas/CAMS/plots/"
#%%
df=pd.read_csv(f"/home/agkiokas/CAMS/plots/{name_station}_{species}_A_30min.csv")
print(df)
print(df.describe())
print(df.columns)
#%%
df1=df.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df1 = df.groupby('timestamp').mean('center_ppb')
print(df1)
# %%
plt.figure(figsize=(10, 5))
plt.plot(df1.index, df1['center_ppb'],label=f'{name_station}')
plt.title(f'Timeseries of {species} for the cell where {name_station} station falls into')
plt.xlabel('Time')
plt.ylabel(f'{species} {units}')
plt.xticks(rotation=45)
plt.legend()
plt.show()
plt.savefig(f"{out_dir}/{name_station}_{species}_timeseries_central_pixel.png", dpi=200)

# %%
df2=df.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df2=df.groupby('sector').mean('center_ppb')
df2
# %%
print(type(df2['date']))
# %%
