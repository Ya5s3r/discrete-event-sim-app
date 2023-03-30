import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="📖",
)

st.write("# Introduction")

st.markdown(
    """
    What you will simulate is basically a simplified hospital urgent care system. Click below to see a
    plan of what this looks like...

    """
)
with st.expander("See plan"):
    st.write("""
        The flowchart depicts the system through which patients will flow.
        We can see elements such as resources (yellow boxes), a decision tree
        (orange diamond) and green ‘sinks’ (entity exit points).
       """)
    st.image("./flowchart.jpg")

st.write("## Stochastic Modelling")

st.markdown(
    """
    We can introduce randomness at certain points in our model. This reflects how patient arrive 
    into our system, or how long it takes for a resource (doctor, nurse) to deal with a patient.
    In our interactive app here, you can adjust this degree of randomness by amending values such as 
    mean interarrival times, which generate exponential distributions, or mean and standard deviation
    to model how long a patient spends with a resource (see detailed blog for more details).
    
    You can therefore adjust the level of demand on this simulated urgent care facility using the *mean interarrival*
    slider and visualise what impact this has on your system!

    """
)

st.write("## Queue Management")

st.markdown(
    """
    The model incorporates both first-in, first-out (FIFO) queuing principles (for nurse triage), as well 
    as priority queues (for doctor or cubicle resources), following prioritisation by the triage nurse. The SimPy 
    package provides a very convenient framework for constructing these queue types.

    """
)

st.write("## Use Case")

st.markdown(
    """
    There are numerous useful applications for this kind of model. Examples include *what if?* scenarios. The model
    allows a user to simulate the impact of operational changes on throughput times and resource pressures in a safe manner prior
    implementation in the real world. You can also adjust resources until you reach a target throughput time for entities (patients).

    """
)