from langchain_openai import AzureChatOpenAI
import os
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from tools import *

load_dotenv()


class DuplicateRecommenderAgent:
    def __init__(self):
        self.model = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini-2",
            api_version="2024-05-01-preview",
            model="gpt-4o",
        )
        agent_tools = [fetch_top_k_duplicate]
        # System prompt
        system_prompt = SystemMessage(
            """
            You are an assistant that helps users evaluate clusters of profiles to identify duplicates. When responding:
            you will use the tools available to fetch the top k cluster group, if user did not specify, k is default to be 1.
            Return all the fields in good table forma, make sure columns are aligned.
            Arrange the table columns in this order Full Name,Email,Phone,Company Name, Role, Permission, Zip Code, Cluster
            Record the cluster id.
            For each cluster group, you will ask
            - Is this/are these actual duplicate user(s)? (Yes/No/Unsure)  
            - Confidence in your answer (0-10)  
            """
        )

        self.agent_graph = create_react_agent(
            model=self.model,
            state_modifier=system_prompt,
            tools=agent_tools,
            debug=False,
        )

    def invoke(self, inputs, config):
        return self.agent_graph.invoke(inputs, config=config)


# **IMPORTANT**: Return the raw results from the tool **exactly as they are**. Do not modify, parse, or interpret the results in any way.
