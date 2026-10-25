import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
import streamlit as st
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title='Dark Store Location Optimiser - Chennai',
    page_icon='🛒',
    layout='wide',
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

st.title('🛒 Dark Store Location Optimisation System (Chennai Region)')
st.markdown(
    'Optimize supply chain network design by selecting optimal land-based dark store locations '
    'across Chennai, allocating customer demand under capacity constraints, and enforcing maximum service distance limits.'
)

# Realistic Land-Based Chennai Hubs (Avoiding the Bay of Bengal to the East)
CHENNAI_LAND_HUBS = [
    ('Anna Nagar', 13.0850, 80.2100),
    ('T. Nagar', 13.0418, 80.2341),
    ('Velachery', 12.9810, 80.2240),
    ('Adyar', 13.0012, 80.2565),
    ('Mylapore', 13.0368, 80.2676),
    ('Guindy', 13.0067, 80.2206),
    ('Porur', 13.0382, 80.1584),
    ('Tambaram', 12.9249, 80.1436),
    ('Sholinganallur', 12.9010, 80.2279),
    ('Madhavaram', 13.1482, 80.2305),
    ('Kodambakkam', 13.0524, 80.2215),
    ('KK Nagar', 13.0337, 80.1983),
    ('Nungambakkam', 13.0569, 80.2425),
    ('Kilpauk', 13.0822, 80.2403),
    ('Perambur', 13.1112, 80.2337),
    ('Thiruvanmiyur', 12.9825, 80.2619),
    ('Pallavaram', 12.9675, 80.1491),
    ('Chromepet', 12.9516, 80.1462),
    ('Vadapalani', 13.0503, 80.2121),
    ('Ashok Nagar', 13.0378, 80.2078),
]

# Sidebar Inputs
with st.sidebar:
  st.header('⚙️ Model Parameters')
  with st.form(key='optimization_form'):
    num_customers = st.number_input(
        'Customer Base Scale (I)',
        min_value=500,
        max_value=20000,
        value=10000,
        step=500,
    )
    num_stores = st.number_input(
        'Candidate Dark Stores (J)', min_value=2, max_value=50, value=50, step=1
    )
    max_distance_km = st.slider(
        'Max Service Distance / Radius (km)',
        min_value=5.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
    )

    submit_button = st.form_submit_button(
        label='🚀 Run Optimization', use_container_width=True
    )


@st.cache_data
def generate_chennai_land_data(n_cust_scale, n_str):
  np.random.seed(42)
  n_clusters = min(300, max(50, n_cust_scale // 30))

  cust_coords = {}
  D = {}
  total_target_demand = n_cust_scale * 20

  for i in range(n_clusters):
    hub = CHENNAI_LAND_HUBS[np.random.choice(len(CHENNAI_LAND_HUBS))]
    lat = hub[1] + np.random.uniform(-0.04, 0.04)
    lon = hub[2] + np.random.uniform(-0.04, 0.04)
    cust_coords[i] = (lat, lon)
    D[i] = int(total_target_demand / n_clusters * np.random.uniform(0.8, 1.2))

  store_coords = {}
  for j in range(n_str):
    hub = CHENNAI_LAND_HUBS[j % len(CHENNAI_LAND_HUBS)]
    lat = hub[1] + np.random.uniform(-0.03, 0.03)
    lon = hub[2] + np.random.uniform(-0.03, 0.03)
    store_coords[j] = (lat, lon)

  avg_fixed_cost = 120000
  F = {
      j: int(avg_fixed_cost * np.random.uniform(0.85, 1.15)) for j in range(n_str)
  }

  total_demand = sum(D.values())
  C = {
      j: int(total_demand * np.random.uniform(0.1, 0.3)) for j in range(n_str)
  }

  d = {}
  for i in range(n_clusters):
    for j in range(n_str):
      lat1, lon1 = cust_coords[i]
      lat2, lon2 = store_coords[j]
      dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.0
      d[(i, j)] = dist

  return cust_coords, store_coords, D, F, C, d, n_clusters


cust_coords, store_coords, D, F, C, d, n_clusters = generate_chennai_land_data(
    num_customers, num_stores
)

TRANSPORT_COST_PER_KM = 15.0
REVENUE_PER_UNIT = 180.0


@st.cache_data
def solve_dark_store_model(
    n_cust, n_str, max_stores_open, custom_max_dist, D_dict, F_dict, C_dict, d_dict
):
  solver = pywraplp.Solver.CreateSolver('CBC')
  if not solver:
    return None

  solver.set_time_limit(30000)

  X = {}
  for i in range(n_cust):
    for j in range(n_str):
      if d_dict[(i, j)] <= custom_max_dist:
        X[(i, j)] = solver.NumVar(0.0, solver.infinity(), f'X_{i}_{j}')

  y = {}
  for j in range(n_str):
    y[j] = solver.IntVar(0, 1, f'y_{j}')

  objective = solver.Objective()
  for (i, j), var in X.items():
    objective.SetCoefficient(var, d_dict[(i, j)] * TRANSPORT_COST_PER_KM)
  for j in range(n_str):
    objective.SetCoefficient(y[j], F_dict[j])
  objective.SetMinimization()

  for i in range(n_cust):
    valid_stores = [j for j in range(n_str) if (i, j) in X]
    if not valid_stores:
      return {'status': 'Infeasible'}
    constraint = solver.Constraint(D_dict[i], D_dict[i], f'Demand_{i}')
    for j in valid_stores:
      constraint.SetCoefficient(X[(i, j)], 1.0)

  for j in range(n_str):
    constraint = solver.Constraint(-solver.infinity(), 0.0, f'Capacity_{j}')
    for i in range(n_cust):
      if (i, j) in X:
        constraint.SetCoefficient(X[(i, j)], 1.0)
    constraint.SetCoefficient(y[j], -C_dict[j])

  if max_stores_open is not None:
    store_count_constraint = solver.Constraint(
        0, max_stores_open, 'Max_Stores_Open_Limit'
    )
    for j in range(n_str):
      store_count_constraint.SetCoefficient(y[j], 1.0)

  status = solver.Solve()

  if status == pywraplp.Solver.OPTIMAL:
    total_cost = solver.Objective().Value()
    open_stores = [j for j in range(n_str) if y[j].solution_value() > 0.5]
    allocations = {
        (i, j): X[(i, j)].solution_value()
        for (i, j) in X
        if X[(i, j)].solution_value() > 1e-5
    }
    return {
        'status': 'Optimal',
        'total_cost': total_cost,
        'open_stores': open_stores,
        'allocations': allocations,
    }
  else:
    return {'status': 'Infeasible'}


with st.spinner('Running optimization model...'):
  results = solve_dark_store_model(
      n_clusters, num_stores, None, max_distance_km, D, F, C, d
  )

if results and results['status'] == 'Optimal':
  total_demand_val = sum(D.values())
  total_revenue = total_demand_val * REVENUE_PER_UNIT
  total_system_cost = results['total_cost']
  max_profit = total_revenue - total_system_cost

  # HUGE PROFIT HIGHLIGHT BOX
  st.markdown(
      f"""
        <div class="profit-card">
            <h2>MAXIMUM NET PROFIT</h2>
            <h1>₹ {max_profit:,.0f}</h1>
        </div>
    """,
      unsafe_allow_html=True,
  )

  tab1, tab2, tab3 = st.tabs(
      [
          '🗺️ Detailed Solution Map',
          '📊 KPI & Allocation Table',
          '📈 Scenario Profit Analysis',
      ]
  )

  with tab1:
    st.subheader(
        'Geospatial Visualisation across Chennai Land Hubs with Store Badges'
    )
    m = folium.Map(location=(13.0827, 80.2707), zoom_start=12)

    for j in range(num_stores):
      is_open = j in results['open_stores']
      assigned_demand_j = sum(
          val for (cust, store), val in results['allocations'].items() if store == j
      )
      utilization_j = (
          (assigned_demand_j / C[j]) * 100 if is_open and C[j] > 0 else 0
      )

      badge_html = f"""
            <div style="background-color: {'#1b5e20' if is_open else '#b71c1c'}; color: white; 
                        padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; 
                        text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.3); white-space: nowrap;">
                {'🟢 Open' if is_open else '🔴 Closed'} #{j} ({utilization_j:.0f}% Cap)
            </div>
            """

      popup_html = f"""
            <div style="font-family: Arial; width: 210px;">
                <h4 style="margin: 0; color: {'#2e7d32' if is_open else '#c62828'};">
                    {'🟢 OPEN STORE' if is_open else '🔴 CLOSED STORE'} #{j}
                </h4>
                <hr style="margin: 5px 0;">
                <b>Fixed Cost:</b> ₹{F[j]:,}<br>
                <b>Capacity:</b> {C[j]} units<br>
                <b>Assigned Demand:</b> {assigned_demand_j:.1f} units<br>
                <b>Utilization:</b> {utilization_j:.1f}%<br>
            </div>
            """

      folium.Marker(
          location=store_coords[j],
          popup=folium.Popup(popup_html, max_width=250),
          tooltip=f"Store {j} ({'Open' if is_open else 'Closed'})",
          icon=folium.DivIcon(
              html=badge_html, icon_size=(100, 30), icon_anchor=(50, 35)
          ),
      ).add_to(m)

      if is_open:
        folium.Circle(
            location=store_coords[j],
            radius=max_distance_km * 1000,
            color='green',
            weight=1,
            fill=True,
            fill_opacity=0.04,
        ).add_to(m)

    for i in range(n_clusters):
      folium.CircleMarker(
          location=cust_coords[i],
          radius=3,
          color='dodgerblue',
          fill=True,
          fill_color='dodgerblue',
          popup=f'<b>Customer Cluster {i}</b><br>Demand: {D[i]} units',
      ).add_to(m)

    link_count = 0
    for (i, j), val in results['allocations'].items():
      if link_count < 200:
        folium.PolyLine(
            locations=[cust_coords[i], store_coords[j]],
            color='gray',
            weight=0.8,
            opacity=0.4,
        ).add_to(m)
        link_count += 1

    st_folium(m, width=1100, height=580)

  with tab2:
    st.subheader('📋 Detailed Solution & KPI Breakdown')

    store_summary = []
    for j in range(num_stores):
      is_open = j in results['open_stores']
      assigned_demand = sum(
          val for (cust, store), val in results['allocations'].items() if store == j
      )
      utilization = (
          (assigned_demand / C[j]) * 100 if is_open and C[j] > 0 else 0
      )
      store_summary.append({
          'Store ID': f'Store {j}',
          'Status': 'Open 🟢' if is_open else 'Closed 🔴',
          'Fixed Cost (₹)': f'₹{F[j]:,}',
          'Capacity': C[j],
          'Assigned Demand': f'{assigned_demand:.1f}',
          'Capacity Utilization (%)': f'{utilization:.1f}%',
      })

    st.dataframe(pd.DataFrame(store_summary), use_container_width=True)

  with tab3:
    st.subheader(
        '📈 Profit Analysis when Up to 2, 5, 10, 20, 25 Dark Stores Open'
    )
    st.markdown(
        'Evaluating network profitability across different restrictions on maximum open dark stores.'
    )

    store_limits = [2, 5, 10, 20, 25]
    profits = []
    costs = []

    for limit in store_limits:
      res_limit = solve_dark_store_model(
          n_clusters, num_stores, limit, max_distance_km, D, F, C, d
      )
      if res_limit and res_limit['status'] == 'Optimal':
        cost = res_limit['total_cost']
        profit = total_revenue - cost
        costs.append(cost)
        profits.append(profit)
      else:
        costs.append(0)
        profits.append(0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        store_limits,
        profits,
        marker='o',
        color='forestgreen',
        linewidth=2.5,
        label='Net Profit (₹)',
    )
    ax1.set_title(
        'Total Profit vs. Maximum Allowed Open Dark Stores',
        fontsize=12,
        weight='bold',
    )
    ax1.set_xlabel('Max Allowed Open Dark Stores', fontsize=10)
    ax1.set_ylabel('Net Profit (₹)', fontsize=10, color='forestgreen')
    ax1.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

    sensitivity_df = pd.DataFrame({
        'Max Stores Allowed': store_limits,
        'Total System Cost (₹)': [
            f'₹{c:,.0f}' if c > 0 else 'Infeasible' for c in costs
        ],
        'Total Net Profit (₹)': [
            f'₹{p:,.0f}' if p > 0 else 'Infeasible' for p in profits
        ],
    })
    st.dataframe(sensitivity_df, use_container_width=True)

else:
    st.error(
        '❌ The model is infeasible with the current parameters! '
        'Please increase the **Max Service Distance** to at least **15 km** in the sidebar.'
    )
