from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def init_database(database: str) -> SQLDatabase:    
    db_uri = f'sqlite:///{database}'

    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
	You are a data analyst at a pokemon training center. You are interacting with a pokemon trainer who is asking you questions about the pokemons's database.
	Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.    

    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.    
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
  # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY, temperature=0)

  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
	You are a data analyst at a pokemon training center. You are interacting with a pokemon trainer who is asking you questions about the pokemons's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.    
	<SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
  llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY, temperature=0)

  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a Pokemon Analyst. Ask me anything about your pokemons."),
    ]

# Database Config
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using SQLite. Connect to the database and start chatting.")
        
    st.text_input("Database", value="pokedex.sqlite", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(                
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

# Seperate AI and human messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)        
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))