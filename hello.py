import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome! 👋")

st.markdown(
    """
    Welcome to a simple interactive demonstration of :red[descrete event simulation]
    modelling using the package :red[SimPy]!
    This app allows you to alter the dynamics of an urgent care facility and visualise
    the impact on patient throughput (waiting times), as well as pressures on resource capacity.
    **👈 Select *simulation demo* from the sidebar** to dive right in!
    ### Want to learn more?
    - Check out my introductory [blog](https://medium.com/@yasser.mushtaq/discrete-event-simulation-simply-put-4ae9f098a809) on DES modelling on Medium 
    - A more detailed [blog](https://medium.com/towards-data-science/object-oriented-discrete-event-simulation-with-simpy-53ad82f5f6e2) on how the simulation was 
    constructed using the SimPy Python package
    - Some basic introductory notes are also included in *Introduction* 👈 
"""
)

with st.sidebar:
    st.markdown("""App developed by [Yasser Mushtaq](https://www.linkedin.com/in/yasser-mushtaq-b1bbaa65/)  """)
    st.markdown("""View code at my [GitHub]('https://github.com/Ya5s3r')""")
    
    #https://icons8.com/icon/12598/github