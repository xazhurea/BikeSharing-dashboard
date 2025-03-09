import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# Load data
def load_data():
    df = pd.read_csv("dashboard/maindata.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.header("Filter Tanggal")
date_range = st.sidebar.date_input("Pilih Rentang Tanggal", (df['dteday'].min(), df['dteday'].max()))
start_date, end_date = date_range

# Filter data berdasarkan tanggal
filtered_df = df[(df['dteday'] >= str(start_date)) & (df['dteday'] <= str(end_date))]

# Informasi Ringkasan
st.title("Dashboard Analisis Penyewaan Sepeda")
col1, col2 = st.columns(2)
col1.metric("Total Penyewaan Sepeda", filtered_df['cnt'].sum())
col2.metric("Rata-rata Penyewaan per Hari", round(filtered_df['cnt'].mean(), 2))

col3, col4 = st.columns(2)
col3.metric("Total Pengguna Terdaftar", filtered_df['registered'].sum())
col4.metric("Rata-rata Pengguna Terdaftar", round(filtered_df['registered'].mean(), 2))

# Pola Peminjaman Berdasarkan Waktu
st.header("Pola Peminjaman Sepeda Berdasarkan Waktu ⏰")
st.write("Analisis ini menunjukkan pola peminjaman sepeda berdasarkan waktu. Selain itu, terdapat analisis mengenai perbedaan pola waktu berdasarkan hari libur serta perbedaan antara pengguna casual dan registered.")

col1, col2 = st.columns(2)
with col1:
    st.write("#### Hari Kerja vs Hari Libur")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='hr', y='cnt', data=filtered_df[filtered_df['workingday'] == 1], estimator='mean', label='Hari Kerja', color='blue', ax=ax)
    sns.lineplot(x='hr', y='cnt', data=filtered_df[filtered_df['workingday'] == 0], estimator='mean', label='Hari Libur', color='orange', ax=ax)
    st.pyplot(fig)

with col2:
    st.write("#### Pengguna Casual vs Registered")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='hr', y='casual', data=filtered_df, estimator='mean', label='Casual', color='orange', ax=ax)
    sns.lineplot(x='hr', y='registered', data=filtered_df, estimator='mean', label='Registered', color='blue', ax=ax)
    st.pyplot(fig)


# Pola Peminjaman Berdasarkan Iklim
st.header("Pola Peminjaman Berdasarkan Iklim")
st.write("Analisis ini mengkaji pengaruh musim, kondisi cuaca, dan kecepatan angin terhadap jumlah peminjaman sepeda.")

st.write("#### Pengaruh Musim terhadap Jumlah Peminjaman Sepeda")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=filtered_df, x='season', y='cnt')
st.pyplot(fig)

st.write("#### Pengaruh Kondisi Cuaca terhadap Jumlah Peminjaman Sepeda")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=filtered_df, x='weathersit', y='cnt')
st.pyplot(fig)

st.write("#### Pengaruh Kecepatan Angin terhadap Peminjaman Sepeda")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='windspeed', y='cnt', data=filtered_df)
st.pyplot(fig)


# Tabs untuk Analisis RFM
st.header("Analisis RFM")
tab1, tab2, tab3 = st.tabs(["Recency", "Frequency", "Monetary"])

rfm_df = filtered_df.groupby('instant').agg({
    'dteday': 'max',  # Recency: Tanggal terakhir peminjaman
    'cnt': ['count', 'sum']  # Frequency: Total transaksi, Monetary: Total penyewaan
})
rfm_df.columns = ['Recency', 'Frequency', 'Monetary']
rfm_df['Recency'] = (pd.to_datetime(df['dteday'].max()) - pd.to_datetime(rfm_df['Recency'])).dt.days

with tab1:
    st.write("### Distribusi Recency")
    st.write("Berikut adalah grafik histogram dari data recency pengguna (seberapa lama sejak transaksi terakhir pengguna)")
    fig, ax = plt.subplots()
    sns.histplot(rfm_df['Recency'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
with tab2:
    st.write("### Distribusi Frequency")
    st.write("Berikut adalah grafik histogram dari data frequency pengguna (jumlah transaksi penyewaan per pengguna)")
    fig, ax = plt.subplots()
    sns.histplot(rfm_df['Frequency'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
   
with tab3:
    st.write("### Distribusi Monetary")
    st.write("Berikut adalah grafik histogram dari data monetary pengguna (total nilai transaksi yang dilakukan oleh pengguna.)")
    fig, ax = plt.subplots()
    sns.histplot(rfm_df['Monetary'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
# Clustering berdasarkan jam
st.header("Clustering Berdasarkan Jam")
st.write("Analisis ini mengelompokkan pola peminjaman sepeda berdasarkan jam dalam sehari.")

# Pilih fitur numerik
clustering_data = filtered_df[['hr', 'cnt', 'temp', 'hum']]

# Standarisasi data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Hierarchical Clustering
linked = linkage(scaled_data, method='ward')

# Plot Dendrogram
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(linked, labels=clustering_data['hr'].astype(str).values)
ax.set_title("Dendrogram Clustering Berdasarkan Jam")
ax.set_xlabel("Jam")
ax.set_ylabel("Jarak")
st.pyplot(fig)
