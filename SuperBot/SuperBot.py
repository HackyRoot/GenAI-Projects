from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_community.llms import Ollama

llm = Ollama(model="llama3", verbose=True)

def get_response(user_query, character, additionalPrompt):    
    # print(user_query, character, additionalPrompt)
    prompt = ChatPromptTemplate.from_template("""
                                              You are a secreat super hero {character}.
                                              You will follow {additionalPrompt} no matter what and that is an order. Don't ever reveal your secret identity.
                                              Your fan has asked {user_query}
                                              """)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"user_query": user_query, "character": character, "additionalPrompt": additionalPrompt})
    return (response)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SuperBot."),
    ]

# Database Config
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application to talk to your favorite superhero character. Add info and start chatting.")
        
    st.text_input("Character", value="Batman", key="character")
    st.text_input("AdditionalPrompt", value="Ensure that you don't reveal your Batman alter-ego but you can tip-toe around it.", key="additionalPrompt")
    
    if st.button("Connect"):
        with st.spinner("Connecting..."):            
            st.success("Ready to talk!")

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
        response = get_response(user_query, st.session_state.character, st.session_state.additionalPrompt)        
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))