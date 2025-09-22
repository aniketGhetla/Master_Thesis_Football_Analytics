# Deep Learning-based Formation Detection and GNN-driven Tactical Decision Support System for Football Analytics ⚽

This repository contains the implementation of my Master's thesis project, which develops a complete football analytics pipeline for tactical analysis and recommendation. The system integrates **tracking and event data**, deep learning models, and knowledge graph reasoning to generate interpretable tactical insights.

---

## Features

- **Data Ingestion & Preprocessing**
  - Unified pipeline for **Sportec** and **Metrica** datasets
  - Synchronization of event & tracking data
  - Normalization into a PostgreSQL database

- **Phase of Play Classification**
  - CNN/ResNet-18 model on image-like player/ball representations
  - Labels: *build-up, attack, high block, mid block, low block*

- **Formation Detection**
  - Gaussian role templates + Shaw ordering for role alignment
  - Wasserstein distance + Hungarian matching for formation clustering
  - ResNet fallback for ambiguous cases

- **Formation-vs-Formation Analysis**
  - Tactical metrics: xG, xT, possession%, recovery time, PPDA, pressing intensity
  - Aggregated matchup statistics across matches

- **Tactical Recommendation System**
  - Passing networks modeled as Graph Neural Networks (PyTorch Geometric)
  - Knowledge graph of formations, phases, and style tags
  - Counter-formation suggestions with confidence scores and coaching notes

- **Interactive Dashboard**
  - Flask + Plotly interface
  - Visualizations: pass maps, pass networks, shot maps, formation timelines, tactical recommendations

---

## Tech Stack

- **Languages**: Python, SQL  
- **Libraries**: PyTorch, PyTorch Geometric, Scikit-learn, Pandas, NumPy  
- **Data Tools**: PostgreSQL, Airflow, Spark, Kloppy, DataBallPy  
- **Visualization**: Plotly, Tableau, Streamlit, Flask  

