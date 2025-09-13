import streamlit as st
col1, col2 = st.columns(2)
with col1:
    st.metric("Temperature", "20 °C")
with col2:
    st.metric("Humidity", "89%")


