from langchain_openai import AzureChatOpenAI
import os
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from tools import *

load_dotenv()


class DataAgent:
    def __init__(self):
        self.model = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini-2",
            api_version="2024-05-01-preview",
            model="gpt-4o",
        )
        agent_tools = [find_person_identity]
        # System prompt
        system_prompt = SystemMessage(
            """
            You are an intelligent agent designed to interact with the person's identity collection and perform read or update queries. 
            Available options for field: company_name,zip_code,phone1,email,role,permission,full_name.
            Available options for mode: exact_match and full_text.
            Your task is to understand user requests, pick the right mode, use the right field, value, and available tools.
            Make sure to tell user which mode we are currently using.
            Make sure to tell user what the other mode is.
            Return all the fields in a good format. Do not modify, or interpret the results in any way.
            For the task which there is no corresponding tool, you should inform user that you cannot perform the task.
            You will handle small talk and greetings by producing professional responses.
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
