# SinglePageTool.py
import streamlit as st
import pandas as pd
import random
import datetime as dt
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    kw_input_UI,
    nl_input_UI,
    gs_input_UI,
    doc_upload_input_UI,
    gs_csv_input_UI,
    configure,
)
from utils import get_best_abstracts, show_output_UI, gs_session_agg, display_results, get_matching_score_text
from utils import get_schedule, create_schedule, convert_df_to_events
from utils import EmbModel

st.set_page_config(
    page_title="IISE Session Scout", # This sets the tab name
    page_icon="🔎", # You can also set a favicon (emoji, URL, or local file path)
    layout="centered", # Can be "centered" or "wide"
    initial_sidebar_state="auto" # Can be "auto", "expanded", or "collapsed"
)

model_name = 'BAAI/bge-small-en-v1.5'

model_name_short = model_name.split('/')[-1]

dataset = 'IISE 2025'
dataset_name_nospace = dataset.replace(" ","")

presentation_filepath = f'./Datasets/{dataset}/{dataset_name_nospace}_Processed.csv'
session_filepath = f'./Datasets/{dataset}/session_times.csv'
embedding_filepath = f'./Datasets/{dataset}/Embeddings/{model_name_short}.pkl'
special_session_filepath = f'./Datasets/{dataset}/special_sessions.csv'
special_session_emb_filepath = f'./Datasets/{dataset}/Embeddings/{model_name_short}-specialSessions.pkl'

# Loading in resources
@st.cache_resource
def import_resources(emb_path, model_name):
    emb = open(emb_path, 'rb') 
    embed = pickle.load(emb)
    emb.close()
    embedding_model = SentenceTransformer(model_name)
    return embed, embedding_model

@st.cache_data
def import_data(pres_path, session_path, df_pre_emb_path):
    #Loading in probabilities array and the lemmatized dataset
    try:
        df = pd.read_csv(pres_path)
        pre_emb_df = pd.read_csv(df_pre_emb_path)
        session_df = pd.read_csv(session_path)
        print("Files Loaded Successfully")
        return df, pre_emb_df, session_df
    except Exception as e:
        print(f"An error occured while loading the dataset: {e}")
        return None, None, None

def process(input_type, output_pres, display_schedule):
    if input_type == "Use My Google Scholar Profile":
        #Getting df from session state
        prompt_str_df = st.session_state.prompt_items['Google Scholar']

        #Initializing empty dataframes
        session_df_append = pd.DataFrame()
        presentation_df_append = pd.DataFrame()
        for index, row in prompt_str_df.iterrows():

            #Getting best abstracts for each title-abstract combination
            pres_results = get_best_abstracts(row['Title'] + '.' + row['Abstract'])

            #Storing related title to display later
            pres_results['related_title'] = row['Title']

            #Merging all results into one dataframe
            presentation_df_append = pd.concat([presentation_df_append, pres_results])

        
            #TODO: FIX AGGREGATION SO IT AGGREGATES FOR BEST PRESENTATION ACROSS ALL QUERIES
            #Aggregating presentations on a session level
            if not output_pres:
                session_results = gs_session_agg(pres_results)
                session_results['related_title'] = row['Title']
                session_df_append = pd.concat([session_df_append, session_results])

        #For each presentation, taking the best cosine similarity score for all user queried papers searched
        pres_df = presentation_df_append.loc[presentation_df_append.groupby('paper_id')['cosine_similarity'].transform('max')
                                == presentation_df_append['cosine_similarity']]
        
        
        if not output_pres:
            scholar_df = session_df_append.loc[session_df_append.groupby('session_id')['cosine_similarity'].transform('max')
                                == session_df_append['cosine_similarity']]
            scholar_df = scholar_df.sort_values(by='cosine_similarity', ascending=False)
            #where is scholar_df used
            st.session_state.pres_df = pres_df
            show_output_UI(pres_df, 
                            session_level_agg = True,
                            calendar = display_schedule,
                            scholar_df=scholar_df)
        
        else: 
            show_output_UI(pres_df, 
                            session_level_agg = False)

    else:
        df = get_best_abstracts(st.session_state.prompt_items[input_type][0])
        show_output_UI(df, 
                       session_level_agg = not output_pres,
                       calendar = display_schedule)


# Storing variables permenantly so that I can filter these for the search feature
# Embedding model does not need to be filtered
st.session_state.embeddings_perm, st.session_state.embedding_model = import_resources(embedding_filepath, model_name)
if st.session_state.embeddings_perm.doChunk:
    pre_emb_path = f'./Datasets/{dataset}/chunked_df.csv'
else:
    pre_emb_path = f'./Datasets/{dataset}/non_chunked_df.csv'

st.session_state.original_df_perm, st.session_state.pre_emb_perm, st.session_state.session_df_perm = import_data(presentation_filepath, session_filepath, pre_emb_path)

# st.write(st.session_state.embeddings_perm.name)
# st.write(st.session_state.embeddings_perm.embedding)
# st.write(st.session_state.pre_emb_perm)
# st.write(st.session_state.original_df_perm)
# st.write(st.session_state.session_df_perm)

# Sidebar filters
with st.sidebar.expander(label="Filter options"):
    filter_special_sessions = st.toggle('Search for special sessions only')

# The below code chunks configure the filtering

# If there are no filters, set datasets to permanent datasets
filters_applied = filter_special_sessions
if not filters_applied:
    st.session_state.original_df = st.session_state.original_df_perm
    st.session_state.pre_emb = st.session_state.pre_emb_perm
    st.session_state.session_df = st.session_state.session_df_perm
    st.session_state.embeddings = st.session_state.embeddings_perm


# def session_df_filter(session_df, original_df, lem_df, embeddings, column_name, bool_filter_value):
#     #Filters dataframe for special sessions only
#     session_df = session_df[session_df[column_name] == bool_filter_value]
#     session_id_list = session_df['session_id']
#     original_df = original_df[original_df['session_id'].isin(session_id_list)]
#     paper_id_list = original_df['paper_id']

#     lem_embed_mask = lem_df['paper_id'].isin(paper_id_list)
#     lem_df = lem_df[lem_embed_mask]
#     embeddings = embeddings[lem_embed_mask]

#     return session_df, original_df, lem_df, embeddings

# st.session_state.session_df, st.session_state.original_df, st.session_state.lem_df, st.session_state.embeddings = session_df_filter(
#                                                                               session_df=st.session_state.session_df_perm,
#                                                                               original_df=st.session_state.original_df_perm,
#                                                                               lem_df=st.session_state.lem_df_perm,
#                                                                               embeddings=st.session_state.embeddings_perm,
#                                                                               column_name='session_type',
#                                                                               bool_filter_value=filter_special_sessions)


if "prompt_items" not in st.session_state:
    st.session_state.prompt_items = {
        "Google Scholar": "",
        "Keyword": "",
        "Natural Language": "",
        "Upload Profile Document": "",
        "Upload Google Scholar CSV": "",
    }

if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

# Creating streamlit selection elements
st.markdown(f"<div style = 'text-align: center; font-size: 28px; font-weight: bold; '> 📚 {dataset} Conference Presentation Discovery</div>",
            unsafe_allow_html=True)
st.markdown("""<div style = 'text-align: center; font-size: 18px'>  📊 Find Presentations and Sessions Most Relevant to Your Area of Interest! 
             <br></div>""",
            unsafe_allow_html=True)
st.markdown("""<div style = 'text-align: center; font-size: 14px'>  Part of a Cal Poly SLO master's thesis by <strong> Tillman Erb </strong> (tierb@calpoly.edu | 
            <a href="https://www.linkedin.com/in/tillman-erb-927874224/">
            LinkedIn</a>) 
            <br> advised by <strong> Dr. Puneet Agarwal </strong> (pagarw05@calpoly.edu | 
            <a href="https://www.linkedin.com/in/puneetag02/">
            LinkedIn</a>)
            <br></div>""",
            unsafe_allow_html=True)
st.markdown("""<div style = 'text-align: center; font-size: 20px'> <strong>
            Please take the time to fill out <a href="https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=2wING578lUSVNx03nMoq5z2ZCJwz-MVAsXJkBUC9XY9UNVFTTTg0R0VFM0lQUFBUMktFTVRVVTJJVi4u">
            this quick, 3-minute survey</a> about your experience with the webapp. </strong>""",
            unsafe_allow_html=True)

with st.expander(f"💡**How to use**"):
    st.markdown("""
    Welcome to Session Scout! To begin, first select how you'd like to search the conference program using the sidebar on the left.
    
    * **Natural Language**: Search using a natural language query
    * **Key Term**: Search using an assortment of concepts or terms
    * **Google Scholar Profile**: Search using the publications listed in your Google Scholar Profile

    Then, select what you'd like to search for.
    
    * **Presentations**: Search for individual presentations relevant to your query
    * **Sessions**: Search for sessions (sets of presentations) relevant to your query
    * **Generate My Schedule**: Generate a schedule for the entire conference based on the best matching sessions to your query
    """)

input_type1 = st.sidebar.segmented_control(
    label="Select input type",
    options=[
        "Key Term",
        "Natural Language",
        "Upload Profile Document",
        "Upload Google Scholar CSV",
    ],
    default="Natural Language",
)

# temporary code to set input type to keyword
if input_type1 == "Key Term":
    input_type = "Keyword"
else:
    input_type = input_type1

# Code to toggle special session search
if filter_special_sessions:

    def get_best_abstracts_modified(string):
        embeddings = st.session_state.special_embeddings.embedding
        session_df = st.session_state.special_session_df
        embedding_model = st.session_state.embedding_model

        def get_best_embedding_modified(string, embeddings = embeddings, lem_df = session_df):
            str_embedding = embedding_model.encode(string)
            str_embedding_reshaped = str_embedding.reshape(1, -1)  # Reshape only once

            # Vectorized cosine similarity calculation
            cosine_similarities = cosine_similarity(np.array(embeddings), str_embedding_reshaped)
            cosine_similarities = cosine_similarities.flatten()  # Flatten to a 1D array

            lem_df['cosine_similarity'] = cosine_similarities

            return lem_df

        sent_df = get_best_embedding_modified(string=string, embeddings = embeddings, lem_df = session_df)
        sent_df = pd.DataFrame(sent_df)

        sent_df.sort_values(by='cosine_similarity', ascending=False, inplace=True)

        return sent_df

    def process_modified(input_type, output_pres, display_schedule):
        if input_type == "Use My Google Scholar Profile":
            # Getting df from session state
            prompt_str_df = st.session_state.prompt_items['Google Scholar']

            # Initializing empty dataframes
            session_df_append = pd.DataFrame()
            presentation_df_append = pd.DataFrame()
            for index, row in prompt_str_df.iterrows():

                # Getting best abstracts for each title-abstract combination
                pres_results = get_best_abstracts_modified(row['Title'] + '.' + row['Abstract'])

                # Storing related title to display later
                pres_results['related_title'] = row['Title']

                # Merging all results into one dataframe
                presentation_df_append = pd.concat([presentation_df_append, pres_results])

            # For each presentation, taking the best cosine similarity score for all user queried papers searched
            pres_df = presentation_df_append.loc[presentation_df_append.groupby('special_session_id')['cosine_similarity'].transform('max')
                                    == presentation_df_append['cosine_similarity']]

            show_output_UI_modified(pres_df, 
                            session_level_agg = False)

        else:
            df = get_best_abstracts_modified(st.session_state.prompt_items[input_type][0])

            show_output_UI_modified(df, 
                        session_level_agg = False)

    def show_output_UI_modified(output_df, session_level_agg = False, calendar = False, scholar_df = None):
        # Google scholar matching and aggregating
        if isinstance(scholar_df, pd.DataFrame):
            try:

                pres_df = output_df.copy()
                output_df = scholar_df
                output_df['matching_level'] = output_df['cosine_similarity'].apply(
                    lambda x: get_matching_score_text(x, score_dict, color = True))

            except Exception as e:
                print(f'Error: {e}')

        header_text = "Matching Sessions"

        st.header(header_text)

        display_results_modified(output_df=output_df, session_level_agg = session_level_agg)

    def display_results_modified(output_df, pres_df = None, session_level_agg = False, n_results = 10):

        output_df = output_df.sort_values(by='cosine_similarity', ascending=False)
        output_df = output_df.head(n_results)
        dict_type = {'Excellent': 0.76,
                    'Great': 0.72,
                    'Good' : 0.68,
                    'Poor': 0.0}
        output_df['start_datetime'] = pd.to_datetime(output_df['start_datetime'])
        output_df['end_datetime'] = pd.to_datetime(output_df['end_datetime'])
        output_df['date'] = output_df['start_datetime'].dt.strftime('%A, %B %d, %Y')
        output_df['start_time'] = output_df['start_datetime'].dt.strftime('%I:%M %p')
        output_df['end_time'] = output_df['end_datetime'].dt.strftime('%I:%M %p')

        i = 0
        for index, row in output_df.iterrows():
            i = i + 1
            st.subheader(f"**Best match #{i}:** {row['session_name']}")
            st.write(f"**Estimated Match Level:** {get_matching_score_text(row['cosine_similarity'], dict_type)}", unsafe_allow_html=True)
            st.write(f"**Special Session Type:** {row['session_type']}")

            st.write(f"**Date:** {row['date']}")
            st.write(f"**Start Time:** {row['start_time']}")
            st.write(f"**End Time:** {row['end_time']}")
            st.write(f"**Room Number:** {row['room_name']}")

            if 'related_title' in output_df.columns:
                st.write(f"Most relevant to:  {row['related_title']}")
            if type(row['special_session_description']) == str:
                with st.expander(f"**Special Session Description:** (if available) {row['special_session_description'][:80]}..."):
                    st.write(f"{row['special_session_description']}")

    st.session_state.search_triggered = False

    @st.cache_data
    def import_special_session_data(spec_session_path):
        # Loading in probabilities array and the lemmatized dataset
        try:
            spec_session_df = pd.read_csv(spec_session_path)
            print("Files Loaded Successfully")
            return spec_session_df
        except Exception as e:
            print(f"An error occured while loading the dataset: {e}")
            return None

    @st.cache_resource
    def import_special_session_emb(emb_path):
        # Loading in probabilities array and the lemmatized dataset
        emb = open(emb_path, 'rb') 
        embed = pickle.load(emb)
        emb.close()
        return embed

    st.session_state.special_session_df = import_special_session_data(special_session_filepath)
    st.session_state.special_embeddings = import_special_session_emb(special_session_emb_filepath)

    display_schedule = False
    output_pres = False
    st.session_state.searchButtonLabel = "Search for Sessions"

    if input_type == "Keyword":
        kw_input_UI()
    elif input_type == "Natural Language":
        nl_input_UI()
    elif input_type == "Upload Profile Document":
        doc_upload_input_UI()
    elif input_type == "Upload Google Scholar CSV":
        gs_csv_input_UI()

    if st.session_state.search_triggered:
        if input_type == "Keyword":
            try:
                if st.session_state.prompt_items["Keyword"][0] != "":
                    process_modified(
                        input_type=input_type,
                        output_pres=output_pres,
                        display_schedule=display_schedule,
                    )
            except Exception:
                pass

        elif input_type in [
            "Natural Language",
            "Upload Profile Document",
            "Upload Google Scholar CSV",
        ]:
            prompt_value = st.session_state.prompt_items[input_type]

            if (
                isinstance(prompt_value, list)
                and len(prompt_value) > 0
                and prompt_value[0] != ""
            ):
                process_modified(
                    input_type=input_type,
                    output_pres=output_pres,
                    display_schedule=display_schedule,
                )

else:
    output_type = st.sidebar.segmented_control(label = "Are you searching for specific presentations or sessions to attend?",
                            options = ["Sessions",
                                        "Presentations"],
                                        default="Sessions")

    if output_type == "Presentations":
        output_pres = True
        display_schedule = False
        st.session_state.searchButtonLabel = "Search for Presentations"
    else:
        output_pres = False
        if st.sidebar.toggle("Generate My Schedule Mode"):
            display_schedule = True
            st.session_state.searchButtonLabel = "Generate Schedule"
        else:
            display_schedule = False
            st.session_state.searchButtonLabel = "Search for Sessions"

    if input_type == "Keyword":
        kw_input_UI()
    elif input_type == "Natural Language":
        nl_input_UI()
    elif input_type == "Upload Profile Document":
        doc_upload_input_UI()
    elif input_type == "Upload Google Scholar CSV":
        gs_csv_input_UI()

    if st.session_state.search_triggered:
        if input_type == "Keyword":
            try:
                if st.session_state.prompt_items["Keyword"][0] != "":
                    process(
                        input_type=input_type,
                        output_pres=output_pres,
                        display_schedule=display_schedule,
                    )
            except Exception:
                pass

        elif input_type in [
            "Natural Language",
            "Upload Profile Document",
            "Upload Google Scholar CSV",
        ]:
            prompt_value = st.session_state.prompt_items[input_type]

            if (
                isinstance(prompt_value, list)
                and len(prompt_value) > 0
                and prompt_value[0] != ""
            ):
                process(
                    input_type=input_type,
                    output_pres=output_pres,
                    display_schedule=display_schedule,
                )
