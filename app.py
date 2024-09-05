import streamlit as st
import home
import dashboard
import dash2

st.set_page_config(page_title="Image colorization",page_icon='🖼️')
st.sidebar.subheader("Home Page")
choice = st.sidebar.selectbox('Choose Dashboard after login',
     ('Home', 'Dashbord 1' , 'Dashboard 2'))
sasi=''
if choice=='Home':
   home.main()
elif choice=='Dashbord 1':
    dashboard.Dashboard()
else:
    dash2.Dashboard()