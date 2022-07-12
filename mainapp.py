import streamlit as st
st.set_page_config(page_title="Big Basket Product Recommendation System",
     page_icon="logo.png",layout="wide")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df2 = pd.read_csv('data_cleaned_1.csv')
df2 = df2[['product','rating','sale_price','market_price','soup']]
df2 = df2[:10000]

@st.cache(ttl=48*3600)
def get_recommendations_2(title,df2):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df2['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    df2 = df2.reset_index()
    indices = pd.Series(df2.index, index=df2['product'])
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim2[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movie_indices


product_list = df2['product'].values


st.title('Big Basket Product Recommendation System')

title_alignment="""
    <style>
    .css-10trblm {
        text-align: center
        }
    .stButton
    {
        text-align:center;
    }
    .stButton .css-1q8dd3e
    {
        padding: 0.75rem 2.75rem;
        background-color: #40ff00;
        color: black;
        font-weight: bold;
    }
    .css-16huue1
    {
        font-size: 18px;
    }
    .css-13sdm1b 
    {
        text-align: center;
    }
    code {
    font-size: 20px;
    background: None;
    color: #40ff00;
    }

    .css-1a32fsj.e19lei0e0
    {
        display: flex;
    justify-content: center;
    }
    </style>
    """
st.markdown(title_alignment, unsafe_allow_html=True)

selected_product = st.selectbox(
    "Type or Select a product from the dropdown",
    product_list
)



if st.button('Predict'):
    st.subheader(f"Recommended Products:")
    recommended_product_names = get_recommendations_2(selected_product,df2)
    # recommended_product_names = recommended_product_names.reset_index().drop(['index'],axis=1)
    recommended_products = df2.iloc[recommended_product_names].reset_index().drop(['soup','index'],axis=1)
    st.dataframe(recommended_products)
