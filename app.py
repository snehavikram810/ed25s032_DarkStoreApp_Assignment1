import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
import streamlit as st
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title="Dark Store Location Optimiser - Chennai",
    page_icon="🛒",
    layout="wide",
)

# Custom CSS styling for a polished UI & Huge Profit Box
st.markdown(
    """
    <style>
    .main { background-color: #f4f6f9; }
    .profit-card {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .profit-card h2 { margin: 0; font-size: 20px; font-weight: 400; opacity: 0.9; }
    .profit-card h1 { margin: 10px 0 0 0; font-size: 42px; font-weight: 700; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🛒 Dark Store Location Optimisation System (Chennai Region)")
st.markdown(
    "Optimize supply chain network design by selecting optimal land-based dark store locations "
    "across Chennai, allocating customer demand under capacity constraints, and enforcing maximum service distance limits."
)

# Realistic Land-Based Chennai Hubs (Avoiding the Bay of Bengal to the East)
CHENNAI_LAND_HUBS = [
    ("Anna Nagar", 13.0850, 80.2100),
    ("T. Nagar", 13.0418,
