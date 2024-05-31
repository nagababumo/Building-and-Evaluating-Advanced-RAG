#!/usr/bin/env python
# coding: utf-8

# # Lesson 1: Advanced RAG Pipeline

# In[1]:


import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()


# In[2]:


from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()


# In[3]:


print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])


# ## Basic RAG pipeline

# In[4]:


from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))


# In[5]:


from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)


# In[6]:


query_engine = index.as_query_engine()


# In[7]:


response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))


# ## Evaluation setup using TruLens

# In[8]:


eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)


# In[9]:


# You can try your own question:
new_question = "What is the right AI job for me?"
eval_questions.append(new_question)


# In[10]:


print(eval_questions)


# In[11]:


from trulens_eval import Tru
tru = Tru()

tru.reset_database()


# For the classroom, we've written some of the code in helper functions inside a utils.py file.  
# - You can view the utils.py file in the file directory by clicking on the "Jupyter" logo at the top of the notebook.
# - In later lessons, you'll get to work directly with the code that's currently wrapped inside these helper functions, to give you more options to customize your RAG pipeline.

# In[12]:


from utils import get_prebuilt_trulens_recorder

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")


# In[13]:


with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)


# In[14]:


records, feedback = tru.get_records_and_feedback(app_ids=[])


# In[15]:


records.head()


# In[16]:


# launches on http://localhost:8501/
tru.run_dashboard()


# ## Advanced RAG pipeline

# ### 1. Sentence Window retrieval

# In[17]:


from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)


# In[18]:


from utils import build_sentence_window_index

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)


# In[19]:


from utils import get_sentence_window_query_engine

sentence_window_engine = get_sentence_window_query_engine(sentence_index)


# In[20]:


window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))


# In[21]:


tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)


# In[22]:


for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))


# In[24]:


tru.get_leaderboard(app_ids=[])


# In[25]:


# launches on http://localhost:8501/
tru.run_dashboard()


# ### 2. Auto-merging retrieval

# In[26]:


from utils import build_automerging_index

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)


# In[27]:


from utils import get_automerging_query_engine

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)


# In[28]:


auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))


# In[29]:


tru.reset_database()

tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")


# In[30]:


for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)


# In[31]:


tru.get_leaderboard(app_ids=[])


# In[32]:


# launches on http://localhost:8501/
tru.run_dashboard()

