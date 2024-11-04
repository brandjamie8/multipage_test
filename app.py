import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Function to set up the page structure
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Visualization", "Calculation", "Clustering"])

    if page == "Data Visualization":
        data_visualization()
    elif page == "Calculation":
        calculation()
    elif page == "Clustering":
        clustering()

# Data Visualization Page
def data_visualization():
    st.title("Data Visualization Page")

    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Dropdowns for filtering
    feature1 = st.selectbox("Select Feature 1 for X-axis", df.columns[:-1])
    feature2 = st.selectbox("Select Feature 2 for Y-axis", df.columns[:-1])

    # Create two charts
    fig1 = px.scatter(df, x=feature1, y="target", title=f"{feature1} vs Target")
    fig2 = px.scatter(df, x=feature2, y="target", title=f"{feature2} vs Target")

    # Display charts side by side
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# Calculation Page
def calculation():
    st.title("Calculation Page")

    # Input fields for variables
    var1 = st.number_input("Enter value for Variable 1", min_value=0.0, max_value=10.0, value=5.0)
    var2 = st.number_input("Enter value for Variable 2", min_value=0.0, max_value=10.0, value=3.0)

    # Simple calculation
    result = var1 * var2

    st.write(f"Result of {var1} * {var2} = {result}")

    # Visualization of the result (example graph)
    result_data = pd.DataFrame({
        "Variable": ["Var1", "Var2", "Result"],
        "Value": [var1, var2, result]
    })

    fig = px.bar(result_data, x="Variable", y="Value", title="Calculation Result")
    st.plotly_chart(fig, use_container_width=True)

# Clustering Page
def clustering():
    st.title("Clustering Page")

    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Dropdowns for selecting features
    feature1 = st.selectbox("Select Feature 1 for Clustering (X-axis)", df.columns[:-1], key="clust_feature1")
    feature2 = st.selectbox("Select Feature 2 for Clustering (Y-axis)", df.columns[:-1], key="clust_feature2")

    # Slider to choose the number of clusters
    n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df[[feature1, feature2]])

    # Scatter plot of clusters
    fig = px.scatter(df, x=feature1, y=feature2, color="Cluster", title=f"{n_clusters}-Means Clustering")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
