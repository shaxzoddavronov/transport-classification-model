# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:00:44 2023

@author: OneComputers
"""
import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px

plt=platform.system()
if plt=='Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title('Transport Images Classification Model')
file=st.file_uploader('Upload image',type=['png','jpeg','gif','svg'])
if file:
    st.image(file)
    img=PILImage.create(file)

    #model
    model=load_learner('transport_model.pkl')
    pred,pred_id,prob=model.predict(img)
    st.success(f"Transport: {pred}")
    st.info(f"Probability: {prob[pred_id]*100:.2f}")

    #plotting
    fig=px.bar(x=model.dls.vocab,y=prob*100)
    st.plotly_chart(fig)    
else:
    pass
