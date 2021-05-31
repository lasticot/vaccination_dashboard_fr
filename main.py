#%%
import copy
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dashboard import compute_all, make_table

st.write('Hello world')
@st.cache
def computations():
    result = compute_all()
    return result

def display_table(result, age=None):
    make_table(result, age)


result = copy.deepcopy(computations())
st.write(make_table(result))

st.pyplot(make_table(result)[0])
st.pyplot(make_table(result)[1])
