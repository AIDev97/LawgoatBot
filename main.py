import openai
import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains import ConversationChain


# Initializing app

app = FastAPI()
openai.api_key = os.environ['OPENAI_API_KEY']

# Defining Conversation Class

class Conversation:
    def __init__(self,llm,conversation):
        self.llm=llm
        self.conversation = conversation

    def __repr__(self) -> str:
        return "Defines a Conversation Chain with an Entity memory attached to it"
    
    def get_response(self,question:str):
        conv = self.conversation.predict(input=question)
        return conv
    
# Initializing ChatOpenAI & ConversationChain class form LangChain

llm= ChatOpenAI(
            openai_api_key=os.environ['OPENAI_API_KEY']
        )

conversation = ConversationChain(
            llm= llm,
            verbose=True,
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE.from_file('prompt.txt',input_variables=['entities','history','input']),
            memory=ConversationEntityMemory(llm=llm)
        )

# initializing Conversation Chain
conversationchain = Conversation(llm,conversation)    

# Routes

@app.get('/')
def get_root():
    return {'response':'successful'}


@app.get('/{question}')
async def get_response(question:str):
    
    response = conversationchain.get_response(question)
    return {'response':response}