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
import io
import re
import time
from pypdf import PdfReader
from docx import Document
import fitz
import pdfplumber

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

# --------------------------------------------------------------------------------------------------
# Embedding class
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

# ---------------------------------------------------------------------------------------------------
# Specific Return Functions
# Function which returns matching words and colors based on score
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

# -------------------------------------------------------------------------------------------------------------------
# Data Processing Functions
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

# Getting the best abstracts for a given string
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

# ---------------------------------------------------------------------------------------------------------------
# Schedule display functions

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

# Function which creates a schedule based on sesion ranks
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

# --------------------------------------------------------------------------------------------------------------------------------
# Text Splitting Functions

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

# -------------------------------------------------------------------------------------------------
# Data aggregation functions
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

# --------------------------------------------------------------------------------------------
# UI functions

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

        # Getting list of publications
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
# ----------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Profile document upload: resume, CV, LinkedIn PDF, DOCX, TXT


def clean_profile_text(text, max_chars=12000):
    """Clean and limit long text before embedding search."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text)).strip()

    if len(text) > max_chars:
        text = text[:max_chars]

    return text


def clean_extracted_pdf_text(text):
    """
    Clean text extracted from PDFs.
    Helps with weird spacing, repeated blank lines, and broken line wrapping.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Join hyphenated line breaks: recommen-\ndation -> recommendation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert many blank lines to max two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def extract_text_from_pdf_with_pymupdf(uploaded_file):
    """
    Extract PDF text using PyMuPDF.
    Usually better than pypdf for resumes and web-generated PDFs.
    """
    if fitz is None:
        return ""

    try:
        file_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        pages_text = []

        for page_number, page in enumerate(doc, start=1):
            # "text" is simple and usually reliable.
            page_text = page.get_text("text")

            if page_text and page_text.strip():
                pages_text.append(f"\n--- Page {page_number} ---\n{page_text}")

        doc.close()

        return clean_extracted_pdf_text("\n".join(pages_text))

    except Exception:
        return ""


def extract_text_from_pdf_with_pdfplumber(uploaded_file):
    """
    Extract PDF text using pdfplumber.
    Good fallback for layout-heavy PDFs.
    """
    if pdfplumber is None:
        return ""

    try:
        uploaded_file.seek(0)

        pages_text = []

        with pdfplumber.open(uploaded_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text(
                    x_tolerance=1, y_tolerance=3, layout=False
                )

                if page_text and page_text.strip():
                    pages_text.append(f"\n--- Page {page_number} ---\n{page_text}")

        return clean_extracted_pdf_text("\n".join(pages_text))

    except Exception:
        return ""


def extract_text_from_pdf_with_pypdf(uploaded_file):
    """
    Extract PDF text using pypdf.
    Basic fallback.
    """
    try:
        uploaded_file.seek(0)

        pdf_reader = PdfReader(uploaded_file)
        pages_text = []

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()

            if page_text and page_text.strip():
                pages_text.append(f"\n--- Page {page_number} ---\n{page_text}")

        return clean_extracted_pdf_text("\n".join(pages_text))

    except Exception:
        return ""


def extract_text_from_pdf(uploaded_file):
    """
    Best-effort PDF extraction:
    1. PyMuPDF
    2. pdfplumber
    3. pypdf

    Returns the best available extracted text.
    """

    if uploaded_file is None:
        return ""

    # Keep track of extraction attempts.
    extraction_results = []

    # 1. Try PyMuPDF
    uploaded_file.seek(0)
    pymupdf_text = extract_text_from_pdf_with_pymupdf(uploaded_file)
    if pymupdf_text:
        extraction_results.append(("PyMuPDF", pymupdf_text))

    # 2. Try pdfplumber
    uploaded_file.seek(0)
    pdfplumber_text = extract_text_from_pdf_with_pdfplumber(uploaded_file)
    if pdfplumber_text:
        extraction_results.append(("pdfplumber", pdfplumber_text))

    # 3. Try pypdf
    uploaded_file.seek(0)
    pypdf_text = extract_text_from_pdf_with_pypdf(uploaded_file)
    if pypdf_text:
        extraction_results.append(("pypdf", pypdf_text))

    if not extraction_results:
        st.sidebar.error(
            "Could not extract text from this PDF. It may be scanned/image-based."
        )
        return ""

    # Pick the extraction with the most text.
    # For prototype this is usually good enough.
    best_tool, best_text = max(extraction_results, key=lambda item: len(item[1]))

    st.sidebar.caption(f"PDF text extracted using: {best_tool}")

    return best_text


def extract_text_from_docx(uploaded_file):
    """Extract text from an uploaded DOCX file."""
    text = ""

    try:
        file_bytes = uploaded_file.read()
        doc = Document(io.BytesIO(file_bytes))

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"

    except Exception as e:
        st.sidebar.error(f"Could not read DOCX: {e}")

    return text.strip()


def extract_text_from_txt(uploaded_file):
    """Extract text from an uploaded TXT file."""
    try:
        return uploaded_file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("latin-1").strip()
    except Exception as e:
        st.sidebar.error(f"Could not read TXT: {e}")
        return ""


def extract_text_from_profile_document(uploaded_file):
    """Route profile document upload to the right parser."""
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    if file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    st.sidebar.warning("Unsupported document type. Upload PDF, DOCX, or TXT.")
    return ""


@st.cache_data
def load_recommendation_taxonomy(
    filepath="./Datasets/Taxonomy/recommendation_terms.csv",
):
    """
    Load skills, topics, and methods from a CSV taxonomy file.

    Expected columns:
    - term
    - category
    - aliases

    aliases should be separated by semicolons.
    """
    try:
        taxonomy_df = pd.read_csv(filepath)
    except Exception as e:
        st.warning(f"Could not load taxonomy file: {e}")
        return pd.DataFrame(columns=["term", "category", "aliases"])

    required_columns = {"term", "category", "aliases"}

    if not required_columns.issubset(set(taxonomy_df.columns)):
        st.warning("Taxonomy file must include term, category, and aliases columns.")
        return pd.DataFrame(columns=["term", "category", "aliases"])

    taxonomy_df["term"] = taxonomy_df["term"].astype(str).str.strip()
    taxonomy_df["category"] = (
        taxonomy_df["category"].astype(str).str.strip().str.lower()
    )
    taxonomy_df["aliases"] = taxonomy_df["aliases"].fillna("").astype(str)

    taxonomy_df = taxonomy_df[taxonomy_df["term"] != ""]

    return taxonomy_df


def extract_rule_based_interests(
    text, taxonomy_path="./Datasets/Taxonomy/recommendation_terms.csv"
):
    """
    Extract skills, topics, and methods using an external taxonomy CSV.
    """
    if not text:
        return {
            "skills": [],
            "topics": [],
            "methods": [],
        }

    taxonomy_df = load_recommendation_taxonomy(taxonomy_path)

    if taxonomy_df.empty:
        return {
            "skills": [],
            "topics": [],
            "methods": [],
        }

    text_lower = text.lower()

    extracted = {
        "skill": [],
        "topic": [],
        "method": [],
    }

    for _, row in taxonomy_df.iterrows():
        term = row["term"]
        category = row["category"]
        aliases = row["aliases"]

        search_terms = [term]

        if aliases:
            search_terms.extend(
                [alias.strip() for alias in aliases.split(";") if alias.strip()]
            )

        found = False

        for search_term in search_terms:
            pattern = r"\b" + re.escape(search_term.lower()) + r"\b"

            if re.search(pattern, text_lower):
                found = True
                break

        if found and category in extracted:
            extracted[category].append(term)

    return {
        "skills": sorted(list(set(extracted["skill"]))),
        "topics": sorted(list(set(extracted["topic"]))),
        "methods": sorted(list(set(extracted["method"]))),
    }


def doc_upload_input_UI():
    """
    Sidebar UI for resume, CV, LinkedIn PDF, project document, DOCX, or TXT upload.
    Keep LinkedIn PDF here instead of making it its own feature.
    """

    st.sidebar.header("Upload Profile Document")

    with st.sidebar.expander("What can I upload?"):
        st.write("""
            Use this option for:
            - Resume or CV
            - LinkedIn PDF
            - Project description
            - Personal bio
            - DOCX or TXT profile text
            """)

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        key="profile_document_upload",
    )


    if "profile_doc_text" not in st.session_state:
        st.session_state.profile_doc_text = ""

    if "profile_doc_interests" not in st.session_state:
        st.session_state.profile_doc_interests = {
            "skills": [],
            "topics": [],
            "methods": [],
        }

    if st.sidebar.button("Extract Profile Document", key="extract_profile_document"):
        if uploaded_file is None:
            st.sidebar.warning("Upload a file first.", icon="⚠️")
            return

        extracted_text = ""

        if uploaded_file is not None:
            extracted_text = extract_text_from_profile_document(uploaded_file)

        combined_text = clean_profile_text(extracted_text)

        if not combined_text:
            st.sidebar.error("No usable text could be extracted.")
            return

        st.session_state.profile_doc_text = combined_text
        st.session_state.profile_doc_interests = extract_rule_based_interests(
            combined_text
        )
        st.sidebar.success("Profile document extracted.")

    if st.session_state.profile_doc_text:
        with st.sidebar.expander("Review extracted interests", expanded=True):
            extracted = st.session_state.profile_doc_interests

            skills_text = st.text_area(
                "Skills",
                value=", ".join(extracted.get("skills", [])),
                key="profile_doc_skills",
                help="Edit these. Separate with commas.",
            )

            topics_text = st.text_area(
                "Topics",
                value=", ".join(extracted.get("topics", [])),
                key="profile_doc_topics",
                help="Edit these. Separate with commas.",
            )

            methods_text = st.text_area(
                "Methods",
                value=", ".join(extracted.get("methods", [])),
                key="profile_doc_methods",
                help="Edit these. Separate with commas.",
            )


        with st.sidebar.expander("View extracted document text"):
            st.text_area(
                "Document text preview",
                value=st.session_state.profile_doc_text[:4000],
                height=250,
                disabled=True,
                key="profile_doc_text_preview",
            )

        skills = [x.strip() for x in skills_text.split(",") if x.strip()]
        topics = [x.strip() for x in topics_text.split(",") if x.strip()]
        methods = [x.strip() for x in methods_text.split(",") if x.strip()]

        final_profile_query = f"""
        User professional background:
        {st.session_state.profile_doc_text}

        Extracted skills:
        {", ".join(skills)}

        Extracted topics:
        {", ".join(topics)}

        Extracted methods:
        {", ".join(methods)}
        """

        final_profile_query = clean_profile_text(final_profile_query)

        st.session_state.prompt_items["Upload Profile Document"] = [final_profile_query]

        if st.sidebar.button(
            f"✅ {st.session_state.searchButtonLabel}", key="button_profile_doc_search"
        ):
            if final_profile_query.strip():
                st.session_state.search_triggered = True
            else:
                st.warning("No profile text available.", icon="⚠️")


# --------------------------------------------------------------------------------------------
# Google Scholar CSV upload + OpenAlex enrichment


def reconstruct_openalex_abstract(abstract_inverted_index):
    """
    OpenAlex abstracts are often returned as an inverted index:
    {word: [positions]}.
    This reconstructs the readable abstract text.
    """
    if not isinstance(abstract_inverted_index, dict):
        return ""

    position_to_word = {}

    for word, positions in abstract_inverted_index.items():
        if isinstance(positions, list):
            for position in positions:
                position_to_word[position] = word

    if not position_to_word:
        return ""

    words = [position_to_word[i] for i in sorted(position_to_word.keys())]
    return " ".join(words)


def normalize_title_for_match(title):
    """Normalize a title for rough matching."""
    if not title:
        return ""

    title = str(title).lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def simple_title_similarity(title_a, title_b):
    """
    Lightweight title match score for prototype.
    Returns Jaccard overlap between normalized title tokens.
    """
    tokens_a = set(normalize_title_for_match(title_a).split())
    tokens_b = set(normalize_title_for_match(title_b).split())

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))


def normalize_author_name(name):
    """Normalize author names for rough matching."""
    if not name:
        return ""

    name = str(name).lower()
    name = re.sub(r"[^a-z\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def parse_csv_authors(authors_text):
    """
    Parse Google Scholar CSV Authors field.

    Example:
    "Agarwal, Puneet; Tang, Junlin; Zhuang, Jun;"
    """
    if not authors_text or pd.isna(authors_text):
        return []

    authors = []

    for author in str(authors_text).split(";"):
        author = normalize_author_name(author)

        if author:
            authors.append(author)

    return authors


def get_openalex_author_names(work):
    """Extract normalized author names from an OpenAlex work."""
    if not isinstance(work, dict):
        return []

    authorships = work.get("authorships", [])
    author_names = []

    for authorship in authorships:
        author = authorship.get("author", {})

        display_name = author.get("display_name", "")
        normalized = normalize_author_name(display_name)

        if normalized:
            author_names.append(normalized)

    return author_names


def author_name_match(csv_author, openalex_author):
    """
    Rough author matching.

    Handles cases like:
    - csv: "agarwal puneet"
    - openalex: "puneet agarwal"

    We mainly care whether last names overlap.
    """
    csv_tokens = set(normalize_author_name(csv_author).split())
    openalex_tokens = set(normalize_author_name(openalex_author).split())

    if not csv_tokens or not openalex_tokens:
        return False

    # Strong signal: any token overlap with length > 2.
    # This usually catches last names.
    overlap = {
        token for token in csv_tokens.intersection(openalex_tokens) if len(token) > 2
    }

    return len(overlap) > 0


def author_overlap_score(csv_authors_text, openalex_work):
    """
    Score how many CSV authors appear in the OpenAlex work.

    Returns a value between 0 and 1.
    """
    csv_authors = parse_csv_authors(csv_authors_text)
    openalex_authors = get_openalex_author_names(openalex_work)

    if not csv_authors or not openalex_authors:
        return 0.0

    matched_count = 0

    for csv_author in csv_authors:
        for openalex_author in openalex_authors:
            if author_name_match(csv_author, openalex_author):
                matched_count += 1
                break

    return matched_count / len(csv_authors)


@st.cache_data(show_spinner=False)
def search_openalex_by_title(title, year=None, authors="", mailto=""):
    """
    Search OpenAlex Works by title and return the best matching work.

    Match score uses:
    - title similarity
    - publication year match
    - author overlap
    """
    if not title:
        return None

    params = {
        "search": title,
        "per-page": 5,
    }

    if mailto:
        params["mailto"] = mailto

    try:
        response = requests.get(
            "https://api.openalex.org/works",
            params=params,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    results = data.get("results", [])

    if not results:
        return None

    best_openalex_work = None
    best_score = 0.0

    for openalex_work in results:
        openalex_title = openalex_work.get("title", "")

        title_score = simple_title_similarity(title, openalex_title)
        author_score = author_overlap_score(authors, openalex_work)

        year_score = 0.0
        if year is not None and str(openalex_work.get("publication_year", "")) == str(
            year
        ):
            year_score = 1.0

        # Weighted matching score.
        # Title is most important, authors are second, year is a small boost.
        total_score = 0.70 * title_score + 0.20 * author_score + 0.10 * year_score

        if total_score > best_score:
            best_score = total_score
            best_openalex_work = openalex_work

    # Minimum confidence threshold.
    # If authors match, allow a little lower title score.
    if best_score < 0.35:
        return None

    return best_openalex_work


def read_google_scholar_csv(uploaded_file):
    """
    Read Google Scholar CSV.
    Expected columns:
    Authors, Title, Publication, Volume, Number, Pages, Year, Publisher
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df.columns = [str(col).strip() for col in df.columns]

    if "Title" not in df.columns:
        st.sidebar.error("This CSV needs a Title column.")
        return pd.DataFrame()

    keep_cols = [
        col
        for col in [
            "Authors",
            "Title",
            "Publication",
            "Year",
            "Publisher",
        ]
        if col in df.columns
    ]

    df = df[keep_cols].copy()
    df = df.dropna(subset=["Title"])
    df["Title"] = df["Title"].astype(str).str.strip()
    df = df[df["Title"] != ""]

    return df


def enrich_google_scholar_df_with_openalex(df, max_papers=15, mailto=""):
    """
    Enrich Google Scholar CSV rows with OpenAlex title and abstract.
    Uses fallback to CSV title if no OpenAlex match is found.
    """
    enriched_rows = []

    rows_to_process = df.head(max_papers)

    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()

    total = len(rows_to_process)

    for i, (_, row) in enumerate(rows_to_process.iterrows(), start=1):
        title = row.get("Title", "")
        year = row.get("Year", None)

        status.write(f"Searching OpenAlex {i}/{total}: {title[:60]}...")

        authors = row.get("Authors", "")

        work = search_openalex_by_title(
            title=title,
            year=year,
            authors=authors,
            mailto=mailto,
        )

        abstract = ""
        openalex_title = ""

        if work:
            openalex_title = work.get("title", "")
            abstract = reconstruct_openalex_abstract(
                work.get("abstract_inverted_index")
            )

        enriched_rows.append(
            {
                "selected": True,
                "csv_title": title,
                "openalex_title": openalex_title,
                "abstract": abstract,
                "authors": row.get("Authors", ""),
                "publication": row.get("Publication", ""),
                "year": row.get("Year", ""),
                "publisher": row.get("Publisher", ""),
                "openalex_found": bool(work),
                "has_abstract": bool(abstract),
            }
        )

        progress.progress(i / total)
        time.sleep(0.1)

    status.empty()
    progress.empty()

    return pd.DataFrame(enriched_rows)


def build_google_scholar_profile_query(enriched_df, extra_goal_text=""):
    """Build final text query from enriched Google Scholar publications."""
    if enriched_df.empty:
        return ""

    text_chunks = []

    for _, row in enriched_df.iterrows():
        if "selected" in row and not bool(row["selected"]):
            continue

        title = row.get("openalex_title") or row.get("csv_title") or ""
        abstract = row.get("abstract") or ""

        parts = []

        if title:
            parts.append(f"Publication title: {title}.")

        if row.get("publication"):
            parts.append(f"Venue or journal: {row.get('publication')}.")

        if row.get("year"):
            parts.append(f"Year: {row.get('year')}.")

        if row.get("authors"):
            parts.append(f"Authors: {row.get('authors')}.")

        if abstract:
            parts.append(f"Abstract: {abstract}.")

        text_chunks.append(" ".join(parts))

    final_text = "\n\n".join(text_chunks)

    if extra_goal_text:
        final_text += f"\n\nUser conference goal: {extra_goal_text}"

    return clean_profile_text(final_text, max_chars=20000)


def gs_csv_input_UI():
    """
    Sidebar UI for Google Scholar CSV upload.
    This replaces the old Selenium/proxy Google Scholar profile scraping flow.
    """

    st.sidebar.header("Upload Google Scholar CSV")

    with st.sidebar.expander("How to get your Google Scholar CSV"):
        st.write("""
            1. Go to your Google Scholar profile.
            2. Select the publications you want to use.
            3. Click Export.
            4. Choose CSV.
            5. Upload that CSV here.

            Expected columns include Authors, Title, Publication, Year, and Publisher.
            """)

    uploaded_csv = st.sidebar.file_uploader(
        "Upload Google Scholar CSV",
        type=["csv"],
        key="google_scholar_csv_upload",
    )

    max_papers = st.sidebar.slider(
        "Max publications to enrich",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="For the prototype, keep this lower so OpenAlex lookup is faster.",
    )

    goal_text = st.sidebar.text_area(
        "Optional: anything you want to prioritize or avoid?",
        height=100,
        placeholder="Example: Prioritize applied AI sessions. Avoid beginner-level talks.",
        key="gs_csv_goal",
    )
    
    mailto = ""

    if "gs_csv_raw_df" not in st.session_state:
        st.session_state.gs_csv_raw_df = pd.DataFrame()

    if "gs_csv_enriched_df" not in st.session_state:
        st.session_state.gs_csv_enriched_df = pd.DataFrame()

    if st.sidebar.button("Read Google Scholar CSV", key="read_gs_csv"):
        if uploaded_csv is None:
            st.sidebar.warning("Upload a Google Scholar CSV first.", icon="⚠️")
            return

        df = read_google_scholar_csv(uploaded_csv)

        if df.empty:
            st.sidebar.error("No valid publications found in the CSV.")
            return

        st.session_state.gs_csv_raw_df = df
        st.sidebar.success(f"Found {len(df)} publications.")

    if not st.session_state.gs_csv_raw_df.empty:
        with st.sidebar.expander("CSV publications preview", expanded=False):
            st.dataframe(
                st.session_state.gs_csv_raw_df.head(max_papers),
                use_container_width=True,
            )

        if st.sidebar.button(
            "Find abstracts with OpenAlex", key="enrich_gs_csv_openalex"
        ):
            with st.spinner("Searching OpenAlex for abstracts..."):
                enriched_df = enrich_google_scholar_df_with_openalex(
                    st.session_state.gs_csv_raw_df,
                    max_papers=max_papers,
                    mailto=mailto,
                )

            st.session_state.gs_csv_enriched_df = enriched_df

            found_count = (
                int(enriched_df["openalex_found"].sum()) if not enriched_df.empty else 0
            )
            abstract_count = (
                int(enriched_df["has_abstract"].sum()) if not enriched_df.empty else 0
            )

            st.sidebar.success(
                f"OpenAlex found {found_count} matches; {abstract_count} had abstracts."
            )

    if not st.session_state.gs_csv_enriched_df.empty:
        st.sidebar.write("Review selected publications:")

        edited_df = st.sidebar.data_editor(
            st.session_state.gs_csv_enriched_df,
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Use",
                    default=True,
                ),
                "csv_title": "CSV Title",
                "openalex_title": "OpenAlex Title",
                "abstract": st.column_config.TextColumn(
                    "Abstract",
                    width="medium",
                ),
                "openalex_found": "Found",
                "has_abstract": "Has Abstract",
            },
            disabled=[
                "csv_title",
                "openalex_title",
                "abstract",
                "authors",
                "publication",
                "year",
                "publisher",
                "openalex_found",
                "has_abstract",
            ],
            hide_index=True,
            key="gs_csv_editor",
        )

        final_query = build_google_scholar_profile_query(
            edited_df
        )

        st.session_state.prompt_items["Upload Google Scholar CSV"] = [final_query]

        with st.sidebar.expander("View final research profile text"):
            st.text_area(
                "Research profile text",
                value=final_query[:5000],
                height=250,
                disabled=True,
                key="gs_csv_final_text_preview",
            )

        if st.sidebar.button(
            f"✅ {st.session_state.searchButtonLabel}", key="button_gs_csv_search"
        ):
            if final_query.strip():
                st.session_state.search_triggered = True
            else:
                st.warning("No research profile text available.", icon="⚠️")

# -------------------------------------------------------------
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
