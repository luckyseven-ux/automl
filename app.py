from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from pandas_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import plotly.express as px

with st.sidebar:
    st.title('Auto streamML')
    pilihan = st.radio('Navigasi', ['Upload', 'Profiling', 'ML', 'Download'])
    st.info('Aplikasi ini mengizinkan Anda untuk membangun otomatisasi ML pipeline menggunakan Streamlit')

if os.path.exists('dataanda.csv'):
    df = pd.read_csv('dataanda.csv')
else:
    df = pd.DataFrame()  # Inisialisasi DataFrame jika belum ada data.

st.write('Apa yang Anda ingin lakukan?')

if pilihan == 'Upload':
    st.title('Upload file Anda untuk modeling!')
    file = st.file_uploader('Upload file Anda di sini !!')
    if file:
        df = pd.read_csv(file)
        df.to_csv('dataanda.csv', index=False)
        st.dataframe(df)

if pilihan == 'Profiling':
    st.title('Otomatisasi eksplorasi data analisis')
    if not df.empty:
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)
    else:
        st.warning('Anda perlu mengunggah file terlebih dahulu.')

if pilihan == 'ML':
    st.title('Pilih Machine Learning Anda')
    if not df.empty:
        target25 = st.selectbox('Pilih target anda ', df.columns)
        if st.button('Latih Model'):
            setup_df = setup(data=df, target=target25)
            setup_df = pull()
            st.info('Ini adalah pengaturan ML experimen')
            st.dataframe(setup_df)  # Display setup information as text
            best_model = compare_models()
            compare_df = pull()
            st.info('Ini adalah model ML nya')
            st.dataframe(compare_df)
            save_model(best_model, 'Model_Best_Model')
        else:
            st.warning('Anda perlu mengunggah file dan memilih target terlebih dahulu.')
