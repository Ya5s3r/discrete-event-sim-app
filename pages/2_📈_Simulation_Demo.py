from statistics import mean
import simpy
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
import streamlit as st

# custom classes
from Lognormal import Lognormal
# sim related classes
from Tracker import Tracker
from AEPatient import AEPatient

# font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
# font_manager.findfont("Gill Sans")

st.set_page_config(page_title="Simulation Demo", page_icon="📈")

st.markdown("# Discrete Event Simulation - Stochastic Modelling of an Urgent Care Facility")

st.write("")

st.markdown(
    """

    This is a demo discrete event simulation (DES) model used to optimise an urgent care system.

"""
)
st.write("")

# class to hold global parameters - used to alter model dynamics
class p:
    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["Simulation Run Settings", "A&E Params", "MIU Params"])
        with tab1:
            # simulation run metrics
            st.markdown("""Set your preferred settings for the simulation run times.""")
            warm_up = st.number_input("Simulation warm up - recommended!",1, None, 120, step=1)
            sim_duration = st.number_input("Simulation duration - minutes (min = 60)",60, None, 480, step=1)
            number_of_runs = st.number_input("Number of times to run the simulation (set to 50 to limit resource use - in reality would be higher.)",None, None, 50, step=1)

        with tab2:
            st.markdown("""Let's first start by modelling **demand** for our service. This can be done by adjusting the interarrival time
                        mean, used to generate an exponential distribution.""")
        # interarrival mean for exponential distribution sampling
            inter = st.number_input("Input mean interarrival time",1, None, 5, step=1)
        
        #st.write("")
        #with st.sidebar:
            st.markdown("""We can also set the level of capacity in our simulated system.""")
            number_docs = st.slider("Number of A&E doctors", 1, 10, 3)
            number_nurses = st.slider("Number of triage nurses", 1, 10, 2)
            ae_cubicles = st.slider("Number of A&E cubicles", 3, 12, 7)

        # mean and stdev for lognormal function which converts to Mu and Sigma used to sample from lognormal distribution
        #with st.sidebar:
            st.markdown("""Model the amount of time an entity (patient) spends with a resource (doctor/nurse). The Simpy
                            simulation *locks* this resource while it is with an entity.""")
            mean_doc_consult = st.slider("Mean time with doctor",15, 45, 30, step=1)
            stdev_doc_consult = st.slider("Standard deviation for time with doctor",3, 15, 10, step=1)
            mean_nurse_triage = st.slider("Mean time with nurse",5, 15, 10, step=1)
            stdev_nurse_triage = st.slider("Standard deviation for time with nurse",3, 15, 5, step=1)
    
        # mean time to wait for an inpatient bed if decide to admit
        #with st.sidebar:
            st.markdown("""If the patient needs to be admitted to hospital, they will need to wait for a bed - model below with mean.""")
            mean_ip_wait = st.number_input("Input mean interarrival time for inpatient bed",10, None, 90, step=1)
    
        with tab3:
            # MIU metrics
            st.markdown("""Here we set the level of capacity in the minor injury unit (MIU)""")
            number_docs_miu = st.slider("Number of MIU doctors", 1, 10, 2)
            number_nurses_miu = 3 #st.slider("Number of MIU nurses", 1, 10, 3) -- not used atm in the sim
            miu_cubicles = st.slider("Number of MIU cubicles", 3, 12, 5)
            st.markdown("""Model the amount of time an entity (patient) spends with an MIU resource (doctor).""")
            mean_doc_consult_miu = st.slider("Mean time with MIU doctor",15, 45, 20, step=1)
            stdev_doc_consult_miu = st.slider("Standard deviation for time with MIU doctor",3, 15, 7, step=1)

            
    # some placeholders used to track wait times for resources
    wait_triage = []
    wait_cubicle = []
    wait_doc = []
    wait_doc_miu = []


# class representing AE model
class AEModel:
    # set up simpy env
    def __init__(self) -> None:
        self.env = simpy.Environment()
        self.patient_counter = 0
        # set docs and cubicles as priority resources - urgent patients get seen first
        self.doc = simpy.PriorityResource(self.env, capacity=p.number_docs)
        self.nurse = simpy.Resource(self.env, capacity=p.number_nurses)
        self.cubicle = simpy.PriorityResource(self.env, capacity=p.ae_cubicles)
        # MIU resources - all FIFO
        self.doc_miu = simpy.Resource(self.env, capacity=p.number_docs_miu)
        self.nurse_miu = simpy.Resource(self.env, capacity=p.number_nurses_miu)
        self.cubicle_miu = simpy.Resource(self.env, capacity=p.miu_cubicles)

    # a method that generates AE arrivals
    def generate_ae_arrivals(self):
        while True:
            # add pat
            self.patient_counter += 1

            # create class of AE patient and give ID
            ae_p = AEPatient(p_id=self.patient_counter)

            # simpy runs the attend ED methods
            self.env.process(self.attend_ae(ae_p))

            # Randomly sample the time to the next patient arriving to ae.  
            # The mean is stored in the g class.
            sampled_interarrival = random.expovariate(1.0 / p.inter)

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_interarrival)

    def attend_ae(self, patient):
        # this is where we define the pathway through AE
        triage_queue_start = self.env.now
        # track numbers waiting at each point
        track.plot_data(self.env.now, 'triage')
        track.plot_data(self.env.now, 'cubicle')
        track.plot_data(self.env.now, 'ae_doc')
        # request a triage nurse
        with self.nurse.request() as req:
            # append env time
            # track.plot_data(env.now)
            # append to current waiters
            track.waiters['triage'].append(patient)

            # freeze until request can be met
            yield req
            # remove from waiter list (FIFO)
            track.waiters['triage'].pop()
            # track.plot_data(env.now)
            triage_queue_end = self.env.now
            
            if self.env.now > p.warm_up:
                p.wait_triage.append(triage_queue_end - triage_queue_start)

            # sample triage time from lognormal
            lognorm = Lognormal(mean=p.mean_nurse_triage, stdev=p.stdev_nurse_triage)
            sampled_triage_duration = lognorm.sample()
            #sampled_triage_duration = random.expovariate(1.0 / p.mean_nurse_triage)
            # assign the patient a priority
            patient.set_priority()
            
            yield self.env.timeout(sampled_triage_duration)

        # sample chance of being sent home or told to wait for doc
        #proceed_to_doc = random.uniform(0,1)
        # alternative way to select choice
        patient.set_triage_outcome()

        if patient.triage_outcome == 'AE':
            cubicle_queue_start = self.env.now

            with self.cubicle.request(priority = patient.priority) as req_cub: # request cubicle before doctor
                # track cubicle
                track.waiters['cubicle'].append(patient)
                yield req_cub
                track.waiters['cubicle'].pop()
                cubicle_queue_end = self.env.now
                # record AE cubicle wait time
                if self.env.now > p.warm_up:
                        p.wait_cubicle.append(cubicle_queue_end - cubicle_queue_start)
                doc_queue_start = self.env.now

            # request doc if greater than chance sent home
                with self.doc.request(priority = patient.priority) as req_doc:
                    track.waiters['ae_doc'].append(patient)
                    yield req_doc
                    track.waiters['ae_doc'].pop()
                    doc_queue_end = self.env.now
                    if self.env.now > p.warm_up:
                        p.wait_doc.append(doc_queue_end - doc_queue_start)
                    # sample consult time from lognormal
                    lognorm = Lognormal(mean=p.mean_doc_consult, stdev=p.stdev_doc_consult)
                    sampled_consult_duration = lognorm.sample()

                    yield self.env.timeout(sampled_consult_duration)
                # below prob of request for IP bed. AE doc released but not cubicle
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.3:
                    patient.admitted = True                    
                    sampled_ip_duration = random.expovariate(1.0 / p.mean_ip_wait) # sample the wait time for an IP bed - exponential dist
                    yield self.env.timeout(sampled_ip_duration)
                # else leave the system
                

        elif patient.triage_outcome == 'MIU':
            miu_attend_start = self.env.now

            with self.cubicle_miu.request() as req_cub:
                yield req_cub
                
                with self.doc_miu.request() as req:
                    yield req

                    miu_doc_queue_end = self.env.now
                    if self.env.now > p.warm_up:
                        p.wait_doc_miu.append(miu_doc_queue_end - miu_attend_start)
                    # sample consult time
                    lognorm = Lognormal(mean=p.mean_doc_consult_miu, stdev=p.stdev_doc_consult_miu)
                    sampled_consult_duration = lognorm.sample()

                    yield self.env.timeout(sampled_consult_duration)
        # else leave the system
        # record time in system
        patient.time_in_system = self.env.now - triage_queue_start
        if self.env.now > p.warm_up:
            df_to_add = pd.DataFrame({"P_ID":[patient.p_id],
                                      "Priority":[patient.priority],
                                      "TriageOutcome":[patient.triage_outcome],
                                      "TimeInSystem":[patient.time_in_system],
                                      "Admitted":[patient.admitted]})
            df_to_add.set_index("P_ID", inplace=True)
            frames = [track.results_df, df_to_add]
            track.results_df = pd.concat(frames)
            

    # method to run sim
    def run(self):
        self.env.process(self.generate_ae_arrivals())
        
        self.env.run(until=p.warm_up + p.sim_duration)
        # print(f"The mean wait for a triage nurse was {mean(p.wait_triage):.1f} minutes")
        # print(f"The mean wait for a AE doctor was {mean(p.wait_doc):.1f} minutes")
        # print(f"The mean wait for a MIU doctor was {mean(p.wait_doc_miu):.1f} minutes")
        # calculate mean waits per priority
        track.mean_priority_wait()
        track.priority_count()
        return mean(p.wait_triage), mean(p.wait_cubicle), mean(p.wait_doc), mean(p.wait_doc_miu)

     

# For the number of runs specified in the g class, create an instance of the
# AEModel class, and call its run method


all_runs_triage_mean = []
all_runs_cubicle_mean = []
all_runs_doc_mean = []
all_runs_miu_doc_mean = []
all_time_in_system = []
all_number_of_patients = []

all_run_time_wait_key = {
    'triage': {},
    'cubicle': {},
    'ae_doc': {},
    'miu_doc': {}
}

all_run_priority_time_in_system = {
    'Priority1': [],
    'Priority2': [],
    'Priority3': [],
    'Priority4': [],
    'Priority5': []
}

all_run_priority_counts = {
    'Priority1': [],
    'Priority2': [],
    'Priority3': [],
    'Priority4': [],
    'Priority5': []
}

with st.spinner('Wait for it...'): # adds a progress spinner

    for run in range(p.number_of_runs):
        #print (f"Run {run} of {p.number_of_runs}")

        track = Tracker(warm_up=p.warm_up)
        my_ae_model = AEModel()
        triage_mean, cubicle_mean , doc_mean, miu_mean = my_ae_model.run()
        all_runs_triage_mean.append(triage_mean)
        all_runs_cubicle_mean.append(cubicle_mean)
        all_runs_doc_mean.append(doc_mean)
        all_runs_miu_doc_mean.append(miu_mean)
        # number of patients served per run
        all_number_of_patients.append(len(track.results_df))
        # tracking number of waiters in key queues through sim
        for k in all_run_time_wait_key:
            for t, w in zip(track.env_time_all, track.waiters_all[k]):
                #print(t, w)
                all_run_time_wait_key[k].setdefault(round(t), [])
                all_run_time_wait_key[k][round(t)].append(w)
            all_run_time_wait_key[k] = dict(sorted(all_run_time_wait_key[k].items())) # sort items
        all_time_in_system.append(mean(track.results_df['TimeInSystem']))
        # get priority wait times
        for i in range(1, 6):           
            all_run_priority_time_in_system["Priority{0}".format(i)].append(track.priority_means["Priority{0}".format(i)])
        # number of patient per priority
        for i in range(1, 6):           
            all_run_priority_counts["Priority{0}".format(i)].append(track.priority_counts["Priority{0}".format(i)])
        #print ()

    # print(f"The average number of patients served by the system was {round(mean(all_number_of_patients))}")
    # print(f"The overall average wait across all runs for a triage nurse was {mean(all_runs_triage_mean):.1f} minutes")
    # print(f"The overall average wait across all runs for a cubicle was {mean(all_runs_cubicle_mean):.1f} minutes")
    # print(f"The overall average wait across all runs for a doctor was {mean(all_runs_doc_mean):.1f} minutes")
    # print(f"The overall average wait across all runs for a MIU doctor was {mean(all_runs_miu_doc_mean):.1f} minutes")
    # print(f"The mean patient time in the system across all runs was {mean(all_time_in_system):.1f} minutes")
    #print(f"The mean patient time in the system across all runs was {mean(list(itertools.chain(*all_time_in_system))):.1f} minutes")

    # layout results in tabs, with main results one one tab and sample data on another
    tab_results, tab_df = st.tabs(["Results", "Patient Data Sample"])

    with tab_results:
        st.write("The average number of patients served by the system was", round(mean(all_number_of_patients)))
        st.write("The overall average wait across all runs for a triage nurse was", round(mean(all_runs_triage_mean), 2), "minutes")
        st.write("The overall average wait across all runs for a cubicle was", round(mean(all_runs_cubicle_mean), 2), "minutes")
        st.write("The overall average wait across all runs for a doctor was", round(mean(all_runs_doc_mean), 2), "minutes")
        st.write("The overall average wait across all runs for a MIU doctor was", round(mean(all_runs_miu_doc_mean), 2), "minutes")
        st.write("The mean patient time in the system across all runs was", round(mean(all_time_in_system),2), "minutes")

        # another way to present results:
        # col1, col2, col3 = st.columns(3)
        # col1.metric("Average number of patients served", round(mean(all_number_of_patients)))
        # col2.metric("Average wait for triage nurse", round(mean(all_runs_triage_mean), 2))
        # col3.metric("Average wait for a cubicle", round(mean(all_runs_cubicle_mean), 2))
        # col1.metric("Average wait for a doctor", round(mean(all_runs_doc_mean), 2))

    # number of patients per priority
    patients_per_priority = []
    for k in all_run_priority_counts:
        patients_per_priority.append(round(mean(all_run_priority_counts[k])))
    #patients_per_priority


    wait_means = {
        'triage': [],
        'cubicle': [],
        'ae_doc': [],
        'miu_doc': []
    }
    # lower quartiles
    wait_lq = {
        'triage': [],
        'cubicle': [],
        'ae_doc': [],
        'miu_doc': []
    }
    # upper quartiles
    wait_uq = {
        'triage': [],
        'cubicle': [],
        'ae_doc': [],
        'miu_doc': []
    }

    for k in all_run_time_wait_key:
        for t in all_run_time_wait_key[k]:
            wait_means[k].append(round(mean(all_run_time_wait_key[k][t]), 2))
            wait_lq[k].append(np.percentile(all_run_time_wait_key[k][t], 25))
            wait_uq[k].append(np.percentile(all_run_time_wait_key[k][t], 75))

    # all_run_time_wait_key
    # wait_means
    # wait_lq
    with tab_results:
        #col1, col2 = st.columns(2)
        #with col1:
        # change the default font family
        plt.rcParams.update({'font.family':'Gill Sans'})
        # dark theme
        plt.style.use('dark_background')
        # reset the plot configurations to default
        #plt.rcdefaults()
        #plt
        figure_1, ax = plt.subplots()
        # Set x axis and y axis labels
        ax.set_xlabel('Time', loc='right')
        ax.set_ylabel('Mean Number of Waiters', loc='top')
        ax.set_title('Mean Number of Patients Waiting per Simulator Time', loc='left')

        # Add spines
        ax.spines["top"].set(visible = False)
        ax.spines["right"].set(visible = False)
        # Add grid and axis labels
        ax.grid(True, color = "lightgrey", ls = ":") 

        # Plot our data (x and y here)
        x_time_triage = list(all_run_time_wait_key['triage'].keys())
        x_time_cubicle = list(all_run_time_wait_key['cubicle'].keys())
        x_time_ae_doc = list(all_run_time_wait_key['ae_doc'].keys())

        ax.plot(x_time_triage, wait_means['triage'], label='Triage')
        ax.fill_between(x_time_triage, wait_lq['triage'], wait_uq['triage'], alpha=.1)
        ax.plot(x_time_cubicle, wait_means['cubicle'], label='Cubicle')
        ax.fill_between(x_time_cubicle, wait_lq['cubicle'], wait_uq['cubicle'], alpha=.1)
        ax.plot(x_time_ae_doc, wait_means['ae_doc'], label='AE Doctor')
        ax.fill_between(x_time_ae_doc, wait_lq['ae_doc'], wait_uq['ae_doc'], alpha=.1)
        # Create and set up a legend
        ax.legend(loc="upper left")
                # Show the figure
        #figure_1.savefig('mean_waiters_fig.png')
        st.pyplot(figure_1)
        # footnote
        st.markdown("""*Above provides a snapshot of demand for selected resources throughout the simulation.
                    The queue for a cubicle and doctor are linked, given a patient needs to be seen by a doctor before a
                     cubicle is released.*
                      *Shown are mean waiters across all runs. Shaded areas represent upper and lower quartiles.* """)
        #with col2:
        figure_2, ax = plt.subplots()
        # Set x axis and y axis labels
        #ax.set_xlabel('Priority')
        ax.set_ylabel('Mean Time In System (Minutes)', fontname="Gill Sans", loc='top')
        ax.set_title('Mean Patient Time in System by Priority', loc='left')
        #plot
        axes_labels = list(all_run_priority_time_in_system.keys())
        bar_heights = [np.nanmean(all_run_priority_time_in_system[k]) for k in all_run_priority_time_in_system]
        # below calculates the standard error for each priority np.std(data, ddof=1) / np.sqrt(np.size(data))
        std_error = [np.nanstd(all_run_priority_time_in_system[k], ddof=1) / np.sqrt(np.size(all_run_priority_time_in_system[k])) 
                    for k in all_run_priority_time_in_system]
        # and below the standard deviations
        stds = [np.nanstd(all_run_priority_time_in_system[k], ddof=1) for k in all_run_priority_time_in_system]

        ax.bar(x=axes_labels, height=bar_heights, ec = "black", 
            yerr = stds,
            lw = .75, 
            color = "#005a9b", 
            zorder = 3, 
            width = 0.75,
            error_kw=dict(ecolor='red'))
        for x, s in zip(range(0, 5), patients_per_priority):
            ax.text(x=x, y=10, s=s, horizontalalignment='center', color='red')
        # Add spines
        ax.spines["top"].set(visible = False)
        ax.spines["right"].set(visible = False)
        # Add grid and axis labels
        ax.grid(True, color = "lightgrey", ls = ":")                                                        
        #figure_2.savefig('mean_time_in_system_priority_fig.png')
        st.pyplot(figure_2)
        st.markdown("""*Mean time in system by priority — red numbers indicate average number of patients associated 
                    with each priority group. Error bars indicative of standard deviation.* """)
st.success('Done!')

# look at the last results_df
with tab_df:
    st.markdown("""Below are examples of data generated by patients passing through our system. We can see the outcome of their 
                triage, priority, amount of time taken to pass through the system and admission status.""")
    st.write(track.results_df)

# example heat map
# import pandas as pd
# import seaborn as sns
# df_mean_waiters = pd.DataFrame({'Time': all_run_time_wait_key['triage'].keys(),
#               'Triage': wait_means['triage'],
#               'Cubicle': wait_means['cubicle'],
#               'AE Doctor': wait_means['ae_doc']})

# df_mean_waiters.set_index('Time', inplace=True)  

# sns.set()

# ax = sns.heatmap(df_mean_waiters.transpose())
# plt.title("Mean Number of Waiters")
# plt.show()


