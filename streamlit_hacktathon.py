
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import time


st.title("Contextual Profile Analysis Tool")

st.subheader("Name")
title = st.text_input('','')

response = requests.post("https://web-production-07d4.up.railway.app/users",{"name" : title})

if response.ok == True:
    with st.spinner('Wait for it...'):
        time.sleep(60)
    st.success('Done!')
    response1 = response.json()
    st.subheader("Relevant Passages and their Sentiments")
    import pandas as pd
    import streamlit as st
    Passages = []
    Sentiments = []
    pos, neg, neu = 0, 0, 0
    for i in response1['articles']:
        Passages.append(i['test'])
        Sentiments.append(i['sentiment'])
        if i['sentiment']>0:
            pos+=1
        elif i['sentiment']<0:
            neg+=1
        else:
            neu+=1
    # Cache the dataframe so it's only loaded once

    def load_data():
        return pd.DataFrame(
            {
                "Passages": Passages,
                "Sentiments": Sentiments,
            }
        )

    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")

    df = load_data()

    # Display the dataframe and allow the user to stretch the dataframe
    # across the full width of the container, based on the checkbox value
    st.dataframe(df, use_container_width=st.session_state.use_container_width)


    st.subheader("Summary/Contextual Information")
    name = response1["summary"]
    Sentiment_class = response1['sentiment_analysis']
    percentage_value = 9

    with st.expander('Summary', expanded=False):
        st.markdown(name, unsafe_allow_html=True)

    st.subheader("Overall Sentiment")

    with st.expander('Sentiment class', expanded=False):
        st.markdown(Sentiment_class, unsafe_allow_html=True)




    st.subheader("Pictorial Representation")
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    
    labels = ['Positive', 'Negative', 'Neutral']
   
    sizes = [pos, neg, neu]
    # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()

    patches, texts, autotexts = ax1.pie(sizes, labels=labels,
                                            autopct='%.0f%%',
                                            textprops={'size': 'smaller'},
                                            shadow=False, radius=0.5)
    st.pyplot(fig1)


# def pred(text):
# 	x_val = tokenizer(text=texts,add_special_tokens=True,max_length=60,truncation=True,padding='max_length',return_tensors='tf',return_token_type_ids=True,return_attention_mask=True,verbose=True)
# validation = new_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
# pred_val = np.argmax(validation,axis = 1)
# if pred_val==0:
#   print("Negative")
# elif pred_val==1:
#   print("Neutral")
# else:
#   print("Positive")

