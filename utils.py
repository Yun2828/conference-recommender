# utils.py
import streamlit as st
import streamlit_calendar
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import random
from datetime import datetime as dt
from scholarly import scholarly
from scholarly import ProxyGenerator
from sklearn.metrics.pairwise import cosine_similarity
import selenium
from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType
import requests
from bs4 import BeautifulSoup
import urllib.parse

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

#--------------------------------------------------------------------------------------------------
#Embedding class
class EmbModel:
    def __init__(self, name, doChunk):
        self.name = name
        self.doChunk = doChunk
        self.embedding = None  # Initialize the embedding attribute to None

    def set_embedding(self, embedding_array):
        """Sets the embedding attribute to the provided ndarray."""
        if isinstance(embedding_array, np.ndarray):
            self.embedding = embedding_array
        else:
            raise TypeError("embedding must be a numpy ndarray.")
        
#---------------------------------------------------------------------------------------------------
#Specific Return Functions
#Function which returns matching words and colors based on score
def get_matching_score_text(score, dict_1, color = False):
    """
    Returns the matching score text with appropriate color highlighting based on the score.
    """
    excellent_clr = "#008000"
    great_clr = "#87ab69"
    good_clr = "#1C9997"
    poor_clr = "#555555"

    if not color:
        if score > dict_1['Excellent']:
            return f"<span style='color:{excellent_clr}'>**Excellent**</span>"
        elif score > dict_1['Great']:
            return f"<span style='color:{great_clr}'>**Great**</span>"
        elif score > dict_1['Good']:
            return f"<span style='color:{good_clr}'>**Good**</span>"
        else:
            return f"<span style='color:#964B00'>**Poor**</span>"
    else:
        if score > dict_1['Excellent']:
            return excellent_clr
        elif score > dict_1['Great']:
            return great_clr
        elif score > dict_1['Good']:
            return good_clr
        else:
            return poor_clr

#-------------------------------------------------------------------------------------------------------------------
#Data Processing Functions
def agg_session(df):
    df = df.groupby(by='session_id').mean('cosine_similarity')
    df = merge_session_times(df)
    df.sort_values(by='cosine_similarity', ascending=False, inplace=True)
    return df

def gs_session_agg(output_df, kw = False):
    score_dict = {'Excellent': 0.78,
                'Great': 0.75,
                'Good' : 0.70,
                'Poor': 0.0} 
    
    pres_df = output_df.copy()
    output_df = agg_session(output_df)
    output_df['matching_level'] = output_df['cosine_similarity'].apply(
        lambda x: get_matching_score_text(x, score_dict, color = True))
    return output_df

def merge_session_times(df):    
    session_df = st.session_state.session_df
    df = pd.merge(df, session_df, how='left', left_on='session_id', right_on='session_id')
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])
    df['date'] = df['start_datetime'].dt.strftime('%A, %B %d, %Y')
    df['start_time'] = df['start_datetime'].dt.strftime('%I:%M %p')
    df['end_time'] = df['end_datetime'].dt.strftime('%I:%M %p')
    return (df)

#Getting the best abstracts for a given string
def get_best_abstracts(string):
    embeddings = st.session_state.embeddings.embedding
    embedding_model = st.session_state.embedding_model
    lem_df = st.session_state.pre_emb
    original_df = st.session_state.original_df

    def get_best_embedding(string, embeddings = embeddings, lem_df = lem_df):
        str_embedding = embedding_model.encode(string)
        str_embedding_reshaped = str_embedding.reshape(1, -1)  # Reshape only once

        # Vectorized cosine similarity calculation
        cosine_similarities = cosine_similarity(np.array(embeddings), str_embedding_reshaped)
        cosine_similarities = cosine_similarities.flatten()  # Flatten to a 1D array

        lem_df['cosine_similarity'] = cosine_similarities

        test_df = lem_df.groupby(by='paper_id')['cosine_similarity'].mean()

        return test_df
    
#Splitting prompt into sentences
    prompt_df = pd.DataFrame()
    prompt_df['text'] = [string]
    if st.session_state.embeddings.doChunk:
        prompt_df = split_into_sentences(prompt_df)

    #Matching embeddings for each sentence
    agg_df = pd.DataFrame()

    #Splitting into sentences but if the model does not require chunking,
    #Then each 'sentence' will actually be the entire body of text
    for sentence in prompt_df['text']:
        sent_df = get_best_embedding(sentence, embeddings = embeddings, lem_df = lem_df)
        sent_df = pd.DataFrame(sent_df)
        sent_df.reset_index(names='paper_id', inplace=True)
        agg_df = pd.concat([agg_df, sent_df])

    #Aggregating
    agg_df = agg_df.groupby(by='paper_id').mean('cosine_similarity')

    #Percentile mean code
    # else:
    #     agg_df = agg_df.groupby(by='cosine_similarity')
    #     agg_df = top_percentile_mean(agg_df, 'cosine_similarity', percentile= 0.6)
    #     agg_df.columns = ['cosine_similarity']

    output_df = agg_df.merge(original_df, left_index=True, right_on = 'paper_id')
    output_df.sort_values(by='cosine_similarity', ascending=False, inplace=True)

    return merge_session_times(output_df)

#---------------------------------------------------------------------------------------------------------------
#Schedule display functions

def get_schedule(df, print_results = True):
        schedule_df = create_schedule(df)
        schedule_df = schedule_df[schedule_df['scheduled']]
        schedule_df = schedule_df.sort_values(by='start_datetime')
        st.session_state.schedule_df = schedule_df
        schedule_dict = convert_df_to_events(schedule_df)
    
        display_schedule(schedule_dict, print_results)

def display_schedule(schedule_dict, print_results = True):
    #Printed results refresh each time but calendar does not
    #if 'schedule_dict' in st.session_state:
        #Try moving this to above the search button, then try using st.rerun to fix the calendar bug.
    
    mode = st.selectbox(
        "Calendar Mode:",
        ("list", "timegrid"),
        key=f"calendar_mode"
    )
    
    calendar_options = {
        "editable": "true",
        "navLinks": "true",
        "selectable": "true",
    }

    if mode == "timegrid":
        calendar_options = {
            **calendar_options,
            "initialView": "timeGridWeek",
            "initialDate": "2025-06-01",
        }

    elif mode == "list":
        calendar_options = {
            **calendar_options,
            "initialDate": "2025-06-01",
            "initialView": "listMonth",
        }

    state = streamlit_calendar.calendar(
        events=schedule_dict,
        options=calendar_options,
        custom_css="""
        .fc-event-past {
            opacity: 0.8;
        }
        .fc-event-time {
            font-style: italic;
        }
        .fc-event-title {
            font-weight: 700;
        }
        .fc-toolbar-title {
            font-size: 2rem;
        }
        """,
        key=None,
    )

    if state.get("eventsSet") is not None:
        st.session_state["events"] = state["eventsSet"]
    
    pres_df = st.session_state.pres_df
    if 'related_title' in st.session_state.schedule_df:
    #related_tile is in st.session_state.schedule_df only if it's an abstract
        related_title = True

        
    else:
        related_title = False

    dict_type = {'Excellent': 0.78,
                'Great': 0.75,
                'Good' : 0.70,
                'Poor': 0.0}       
        
    if print_results:
        for index, row in st.session_state.schedule_df.iterrows():
            st.subheader(f"**Session**: {row['session_name']}")
            st.write(f"Estimated Match Level: {get_matching_score_text(row['cosine_similarity'], dict_type)}", unsafe_allow_html=True)
            st.write(f"**Date:** {row['date']}")
            st.write(f"**Start Time:** {row['start_time']}")
            st.write(f"**End Time:** {row['end_time']}")
            st.write(f"**Room Number:** {row['room_name']}")
            #st.write(f"Cosine Similarity Score: {row['cosine_similarity']}") #debugging
            if related_title:
                st.write(f"Relevant to: {row['related_title']}")

            #Getting presentations for given session
            session_pres = pres_df[pres_df['session_id'] == row['session_id']]

            with st.expander(f"**Presentations:**"):
                j = 1
                for windex, prow in session_pres.iterrows():
                    st.markdown(f"<h5>Presentation #{j}: {prow['paper_title']}</h5>",
                                unsafe_allow_html=True)
                    st.write(f"Estimated Match Level: {get_matching_score_text(prow['cosine_similarity'], dict_type)}", 
                            unsafe_allow_html=True)
                    if type(prow['prim_audience']) == str:
                        st.write(f"**Primary Intended Audience:** {prow['prim_audience']}")
                    st.write(f"**Abstract**: {prow['abstract']}")
                    j = j + 1

#Function which creates a schedule based on sesion ranks
def create_schedule(df):
  """
  Creates a non-overlapping schedule based on a score.

  Args:
    df: Pandas DataFrame with 'start_datetime', 'end_datetime', and 'score' columns.

  Returns:
    DataFrame with an additional 'scheduled' column (boolean).
  """

  df['scheduled'] = False  # Initialize 'scheduled' column
  scheduled_events = []

  df_sorted = df.sort_values('cosine_similarity', ascending=False)

  for index, row in df_sorted.iterrows():
    current_start = row['start_datetime']
    current_end = row['end_datetime']
    overlap = False

    for event in scheduled_events:
      event_start = event['start_datetime']
      event_end = event['end_datetime']
      if current_start < event_end and current_end > event_start:
        overlap = True
        break

    if not overlap:
      scheduled_events.append(row) 
      df.loc[index, 'scheduled'] = True

  return df

def convert_df_to_events(df):
  """
  Converts a pandas DataFrame with session information to a list of event dictionaries.

  Args:
    df: pandas DataFrame with columns 'session_name', 'start_datetime', and 'end_datetime'.

  Returns:
    A list of dictionaries, where each dictionary represents an event with 'title', 'start', and 'end' keys.
  """
  events = []

  for index, row in df.iterrows():
    event = {
        "title": row['session_name'],
        "start": row['start_datetime'].strftime("%Y-%m-%dT%H:%M:%S"),
        "end": row['end_datetime'].strftime("%Y-%m-%dT%H:%M:%S"),
        "color": row['matching_level'],
    }
    events.append(event)
  return events

#--------------------------------------------------------------------------------------------------------------------------------
#Text Splitting Functions

def split_into_sentences(df):
  """
  Splits a DataFrame with a single row containing a body of text into 
  multiple rows, one per sentence.

  Args:
    df: A pandas DataFrame with a single row and a single column containing text.

  Returns:
    A new pandas DataFrame with one row per sentence.
  """
  text = df.iloc[0, 0]  # Extract the text from the DataFrame
  sentences = nltk.sent_tokenize(text)  # Split the text into sentences
  return pd.DataFrame(sentences, columns=['text'])

#-------------------------------------------------------------------------------------------------
#Data aggregation functions
def top_percentile_mean(grouped_df, column_name, percentile=0.6):
    """
    Calculates the mean of values in the top percentile for each group in a pandas groupby object.

    Args:
        grouped_df: A pandas groupby object.
        column_name: The name of the column to calculate the mean from.
        percentile: The percentile to use for filtering values. Defaults to 0.6 (60th percentile).

    Returns:
        A pandas Series with the mean of the top percentile values for each group.
    """

    def top_mean(group):
        threshold = group[column_name].quantile(percentile)
        return group[group[column_name] >= threshold][column_name].mean()

    return pd.DataFrame(grouped_df.apply(top_mean))

#--------------------------------------------------------------------------------------------
#UI functions

def configure(option): #DEPRECATED
    if option == "Keyword":
        st.session_state.search_triggered_kw = True
        st.session_state.search_triggered_nl = False
        st.session_state.search_triggered_gs = False
    elif option == "Natural Language":
        st.session_state.search_triggered_kw = False
        st.session_state.search_triggered_nl = True
        st.session_state.search_triggered_gs = False
    elif option == "Google Scholar":
        st.session_state.search_triggered_kw = False
        st.session_state.search_triggered_nl = False
        st.session_state.search_triggered_gs = True

def kw_input_UI():
    """Displays keyword input UI elements in the sidebar."""

    st.sidebar.subheader("🔬 Select Key Terms")

    if "mytsks" not in st.session_state:
        st.session_state.mytsks = []

    if "tskclk" not in st.session_state:
        st.session_state.tskclk = []

    def cmpltTask(task):
        idx = st.session_state.mytsks.index(task)
        del st.session_state.mytsks[idx]

    def listTasks():
        st.session_state.tskclk = []
        for i, task in enumerate(st.session_state.mytsks):
            st.session_state.tskclk.append(
                st.sidebar.button(task, 
                                key='l' + f'{dt.now():%d%m%Y%H%M%S%f}' + str(random.random()), 
                                on_click=cmpltTask, 
                                args=(task,), 
                                help='Click to remove key term')
            )

    def get_unique(lst):
        unique_list = []
        for item in lst:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list

    def add_keyword():
        tsk = st.session_state.keyword_input
        if tsk != "":
            st.session_state.mytsks.append(tsk)
            st.session_state.mytsks = get_unique(st.session_state.mytsks)
            st.session_state.keyword_input = ""  # Clear the text input field

    with st.sidebar.expander(f"💡**Key Term Input: How to Use**"):
        st.write("""Input each search term one at a time. For example, if you're interested in operations research 
                     and 3D printing, add 'operations research' to your key term bank, then add '3D printing', and click
                     'Search for presentations!' """)

    st.sidebar.markdown("""
        **Methodology:**

        * Simulation
        * Optimization
        * Operations Research
        * Machine Learning
        """)
    st.sidebar.markdown("""
        **Application:**

        * Sustainability
        * Supply Chain
        * Healthcare
        * Agriculture
        """)

    st.sidebar.markdown("💡 Try any combination of the above, or input your own!")
    st.sidebar.text_input('Enter your key terms', value="", placeholder='Enter a key term', on_change=add_keyword, key="keyword_input")
    if st.sidebar.button('Add Key Term'):
        add_keyword()

    st.sidebar.subheader("Your selected key terms (click to delete):")
    listTasks()

    # Data processing
    str_list = [str.lower() for str in st.session_state.mytsks]
    prompt_str_kw = ', '.join(str_list)
    if len(str_list) > 0:
        st.session_state.prompt_items["Keyword"] = [prompt_str_kw]

    st.sidebar.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #007bff; /* Blue background color */
            color: white; 
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #0056b3; /* Darker blue on hover */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button(f"✅ {st.session_state.searchButtonLabel}", key='button_1'):
        if len(str_list) > 0:
            st.session_state.search_triggered = True
        else:
            st.warning("Select at least one keyword!", icon="⚠️")

def nl_input_UI():
    """Displays natural language input UI elements in the sidebar."""

    st.sidebar.header("Input natural language prompt")
    st.sidebar.write("💡 Try something like \"advancements in sustainable manufacturing processes\", or copy-paste the abstract of a paper!")
    prompt_str = st.sidebar.text_area(label="Enter natural language search here.", height=200)
    st.session_state.prompt_items["Natural Language"] = [prompt_str]

    if st.sidebar.button(f'✅ {st.session_state.searchButtonLabel}', key='button_2'):
        st.session_state.search_triggered = True


def gs_input_UI():
    """Displays Google Scholar input UI elements in the sidebar."""

    @st.cache_resource
    def search_query(query, institution):

        def build_scholar_url(query, institution):
            """Constructs the Google Scholar URL."""
            base_url = "https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q="
            author_parts = []
            if query:
                words = query.split()
                for word in words:
                    author_parts.append(urllib.parse.quote_plus(word) + "+")
            if institution:
                words = institution.split()
                for word in words:
                    author_parts.append(urllib.parse.quote_plus(word) + "+")

            author_string = "".join(author_parts)
            author_string = author_string[:-1]
            final_url = f"{base_url}{author_string}&btnG="
            return final_url


        username = 'INSERT_USER'
        password = 'INSERT_PASSWORD'
        proxy_host = 'pr.oxylabs.io'
        proxy_port = 7777

        proxy_url_with_auth = f'http://customer-{username}:{password}@{proxy_host}:{proxy_port}'

        # Configure Selenium Proxy
        proxy = Proxy()
        proxy.proxy_type = ProxyType.MANUAL
        proxy.http_proxy = proxy_url_with_auth
        proxy.ssl_proxy = proxy_url_with_auth # For HTTPS traffic

        # Add capabilities to Chrome (or your browser of choice)
        options = webdriver.ChromeOptions() # Or FirefoxOptions, EdgeOptions, etc.
        options.proxy = proxy
        options.headless = True

        # For Chrome, it's often more reliable to set the proxy via command-line arguments
        options.add_argument(f'--proxy-server={proxy_url_with_auth}')
        options.add_argument("--headless=new")

        # Initialize the WebDriver with the options
        driver = webdriver.Chrome(options=options)

        site_url = build_scholar_url(query=query, institution=institution)

        driver.get(site_url)
        driver.implicitly_wait(0.5)
        html_content = driver.page_source

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all div elements with the specified class
        results_divs = soup.find_all('div', class_='gs_r')

        extracted_data = []

        for div in results_divs:
            # Extract text from class "gs_rt"
            title_tag = div.find(class_='gs_rt2')
            title = ""  # This variable is initialized but not used in the snippet provided for 'link'
            link = ""   # Initialize link to an empty string
            if title_tag:
                link_tag = title_tag.find('a')
                if link_tag:
                    link = link_tag.get('href')
                    extracted_data.append(link)

        profile_link = "https://scholar.google.com" + extracted_data[0]

        #Getting list of publications
        driver.get(profile_link)

        html_content = driver.page_source

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all div elements with the specified class
        results_divs = soup.find_all(class_='gsc_a_tr')

        publication_list = []

        for div in results_divs:
            # Extract text from class "gs_rt"
            title_tag_pub = div.find(class_='gsc_a_t')
            title_pub = ""  # This variable is initialized but not used in the snippet provided for 'link'
            link_pub = ""   # Initialize link to an empty string
            if title_tag_pub:
                link_tag_pub = title_tag_pub.find('a')
                if link_tag_pub:
                    link_pub = link_tag_pub.get('href')
                    extracted_data.append(link_pub)
                    title_pub = link_tag_pub.get_text(strip=True)
            
            if title_pub and link_pub: # Add to list if at least one field was found
                publication_list.append({
                    "pub_title": title_pub,
                    "pub_link": link_pub
                })
        return publication_list, driver
    
    def get_publication_text(selected_publications, driver):
        base_url = 'https://scholar.google.com'
        modified_links = []

        for item in selected_publications:
            if 'pub_link' in item:  # Check if the key exists
                full_link = base_url + item['pub_link']
                modified_links.append(full_link)

        publication_data = []

        for link in modified_links:
            driver.get(link)
            pub_html = driver.page_source
            pub_soup = BeautifulSoup(pub_html, 'html.parser')
            pub_page_title = ""
            abstract_divs = ""

            title_divs = pub_soup.find(id='gsc_oci_title')
            if title_divs:
                pub_page_title_html = title_divs.find('a')
                if pub_page_title_html:
                    pub_page_title = pub_page_title_html.get_text(strip=True)
            
            abstract_divs = pub_soup.find(id='gsc_oci_descr')
            if abstract_divs:
                abstract_text = abstract_divs.get_text(strip=True)

            concat_text = pub_page_title + ": " + abstract_text
            publication_data.append({'Title': pub_page_title, 'Abstract': abstract_text})
        
        return publication_data
        
    st.sidebar.title("Author Search")

    # Get the search query from the user
    query = st.sidebar.text_input("Search author")
    institution = st.sidebar.text_input("Add institution (required)")

    if query and institution:
        try:
            # Search for authors, keeping driver object
            publication_list, current_driver = search_query(query=query, institution=institution)

            publication_titles = [item["pub_title"] for item in publication_list]

            titles_to_search = st.sidebar.multiselect(label="Select Publications to Search With", options=publication_titles)

            selected_publications = [
                publication for publication in publication_list if publication["pub_title"] in titles_to_search
            ]


            pub_data = get_publication_text(selected_publications=selected_publications, driver=current_driver)

            st.session_state.prompt_items['Google Scholar'] = pd.DataFrame(pub_data)

            if st.sidebar.button(f"✅ {st.session_state.searchButtonLabel}", key='button_3'):
                if pub_data:
                    st.session_state.search_triggered = True
                else:
                    st.warning("Select at least one abstract!", icon="⚠️")
        except:
            st.write('Unable to retrieve author publications. Try ensuring the author name and institution are spelled correctly.')
#----------------------------------------------------------------------------

def show_output_UI(output_df, session_level_agg = False, calendar = False, scholar_df = None):
    score_dict = {'Excellent': 0.78,
                'Great': 0.75,
                'Good' : 0.70,
                'Poor': 0.0}
    
    #Google scholar matching and aggregating
    if isinstance(scholar_df, pd.DataFrame):
        try:

            pres_df = output_df.copy()
            output_df = scholar_df
            output_df['matching_level'] = output_df['cosine_similarity'].apply(
                lambda x: get_matching_score_text(x, score_dict, color = True))

        except Exception as e:
            print(f'Error: {e}')
    
    else:
        if session_level_agg:
            pres_df = output_df.copy()
            #Used for creating schedule later on
            st.session_state.pres_df = pres_df

            output_df = agg_session(output_df)

            output_df['matching_level'] = output_df['cosine_similarity'].apply(
                lambda x: get_matching_score_text(x, score_dict, color = True))
        
        output_df = pd.merge(output_df, 
                            st.session_state.session_df, 
                            left_on='session_id', 
                            right_on='session_id',
                            suffixes=['', '_extra'])

    if session_level_agg:
        header_text = "Matching Sessions"
    else:
        header_text = "Matching Presentations"

    st.header(header_text)
    
    if not calendar:
        if not session_level_agg:
            pres_df = None
        display_results(output_df=output_df, pres_df=pres_df, session_level_agg = session_level_agg)
    else:
        get_schedule(output_df)

def display_results(output_df, pres_df = None, session_level_agg = False, n_results = 10):
    output_df = output_df.sort_values(by='cosine_similarity', ascending=False)
    output_df = output_df.head(n_results)
    dict_type = {'Excellent': 0.76,
                'Great': 0.72,
                'Good' : 0.68,
                'Poor': 0.0}
    i = 0
    for index, row in output_df.iterrows():
        i = i + 1
        if session_level_agg:
            st.subheader(f"**Best match #{i}:** {row['session_name']}")
            st.write(f"**Estimated Match Level:** {get_matching_score_text(row['cosine_similarity'], dict_type)}", unsafe_allow_html=True)
        else:
            st.subheader(f"**Best match #{i}:** {row['paper_title']}")
            st.write(f"**Estimated Match Level:** {get_matching_score_text(row['cosine_similarity'], dict_type)}", unsafe_allow_html=True)
            st.write(f"**Session:** {row['session_name']}")

        st.write(f"**Date:** {row['date']}")
        st.write(f"**Start Time:** {row['start_time']}")
        st.write(f"**End Time:** {row['end_time']}")
        st.write(f"**Room Number:** {row['room_name']}")

        # st.write(f"Cosine Similarity Score: {row['cosine_similarity']}") #Debugging
        if not session_level_agg:
            if type(row['prim_audience']) == str:
                st.write(f"**Primary Intended Audience:** {row['prim_audience']}")
            if 'related_title' in output_df.columns:
                        st.write(f"Most relevant to:  {row['related_title']}")
            with st.expander(f"**Abstract:** {row['abstract'][:90]}..."):
                st.write(f"{row['abstract']}")
        else:
            session_pres = pres_df[pres_df['session_id'] == row['session_id']]
            with st.expander(f"**Presentations:**"):
                j = 1
                for windex, prow in session_pres.iterrows():
                    st.markdown(f"<h5>Presentation #{j}: {prow['paper_title']}</h5>",
                                unsafe_allow_html=True)
                    st.write(f"Estimated Match Level: {get_matching_score_text(prow['cosine_similarity'], dict_type)}", 
                            unsafe_allow_html=True)
                    # st.write(f"Cosine Similarity Score: {prow['cosine_similarity']}")
                    if type(prow['prim_audience']) == str:
                        st.write(f"**Primary Intended Audience:** {prow['prim_audience']}")
                    if 'related_title' in session_pres.columns:
                        st.write(f"Most relevant to:  {prow['related_title']}")
                    st.write(f"**Abstract**: {prow['abstract']}")
                    j = j + 1
                    