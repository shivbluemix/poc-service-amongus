from langchain_openai import AzureChatOpenAI
import os
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from tools import *

load_dotenv()


class FeedbackAgent:
    def __init__(self):
        self.model = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini-2",
            api_version="2024-05-01-preview",
            model="gpt-4o",
        )
        agent_tools = [save_human_feedback]
        # System prompt
        system_prompt = SystemMessage(
            """
            You are an intelligent agent designed to and use tools to 
            record the user's feedback (yes/no/unsure), confience socre a number from 1 to 10, 
            and the cluster id.  
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
