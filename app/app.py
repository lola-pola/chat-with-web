from datetime import datetime, timedelta
from streamlit_chat import message
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import numpy as np
import requests
import openai
import os


# Define Streamlit app


st.set_page_config(page_title="OpenAI Simple Chatbot", page_icon=":robot:")
st.title("OpenAI Simple Chatbot")
with st.sidebar:
        runner = False
        model = st.text_input("model", "text-embedding-ada-002", key="text-embedding-ada-002")
        
        # Set up OpenAI API key
        openai.api_type = "azure"
        openai.api_base = st.text_input("API base", value="https://yoni123.openai.azure.com/", key="api_base")
        openai.api_version = st.text_input("api version",value="2023-05-15")
        openai.api_key = st.text_input('azure openai key', key="KEY_AZURE_AI", value=None, type="password") 

        web_site = st.selectbox("Select a website", ['techcrunch', "The Verge", "The New York Times"],key='web_site')

        days = st.slider("Select a date", 0,30,7,key='days')
        batch_size = st.slider("Select a batch size", 100,500,200,key='batch_size')
        _articles = st.checkbox("Get Articles && Embeddings ", key='get_articles')


        if st.checkbox("Submit", key="submit"):
            st.success("Submitted!")
            runner = True




def techcrunch_runner(web_site,_articles=False,days=1,batch_size=100):

    def get_urls(date,web_site):
        url = f'https://{web_site}.com/' + date.strftime('%Y/%m/%d')
        content = requests.get(url).text
        return [a['href'] for a in BeautifulSoup(content).find_all(
            'a',
            {'class': 'post-block__title__link'}
        )]
        
    def get_article(url):
        content = requests.get(url).text
        article = BeautifulSoup(content).find_all('div', {'class': 'article-content'})[0]
        return [p.text for p in article.find_all('p', recursive=False)]
        

    if _articles:
            urls = sum([get_urls(datetime.now() - timedelta(days=i),web_site) for i in range(days)], [])
            articles = pd.DataFrame({
                'url': urls,
                'article': [get_article(url) for url in urls]
            })
            paragraphs = (
                articles.explode('article')
                .rename(columns={'article': 'paragraph'})
            )
            paragraphs = paragraphs[paragraphs['paragraph'].str.split().map(len) > 10]
            
            embeddings = []
            for i in range(0, len(paragraphs), batch_size):
                embeddings += get_embedding(paragraphs.iloc[i:i+batch_size]['paragraph'], model)
            paragraphs['embedding'] = embeddings
            st.success(f"{len(paragraphs)} pages was embedded")
            st.expander("Show embeddings").dataframe(paragraphs)
            return paragraphs







def get_embedding(texts, model):
    embeddings_data = []
    texts = [text.replace('\n', ' ') for text in texts]
    for text in texts:
        res = openai.Embedding.create(input=text, engine=model)
        embeddings_data.append(res['data'][0]['embedding']) 
    return embeddings_data


def chatterbot(data_paragraphs,query):
    query_embedding = get_embedding([query], model)[0]
    best_idx = data_paragraphs['embedding'].map(
        lambda emb: np.dot(emb, query_embedding) / (
            np.linalg.norm(emb) * np.linalg.norm(query_embedding)
        )
    ).argmax()
    best_paragraph = data_paragraphs.iloc[best_idx]['paragraph']
    return best_paragraph


st.session_state['generated'] = []
st.session_state['past'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    

if runner:
    if web_site == 'techcrunch':
        print(f'{web_site}{_articles}{days}{batch_size}')
        with st.spinner(f"Getting data from {web_site} from the last {days}"):
            data_paragraphs = techcrunch_runner(web_site=web_site,_articles=_articles,days=days,batch_size=batch_size)
    user_input=st.text_input("You:",key='input')
    if user_input:
        output=chatterbot(data_paragraphs,user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user') 
                st.session_state.generated = ''

