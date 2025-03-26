from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)
import operator
import functools
from langchain_openai import AzureChatOpenAI
from data_agent import DataAgent
from duplicate_recommender_agent import DuplicateRecommenderAgent
from feedback_agent import FeedbackAgent
import uuid


class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# Helper function to invoke an agent
def agent_node(state, agent):
    final_result = None
    # extract thread-id from request for conversation memory
    thread_id = uuid.uuid4()
    # Set the config for calling the agent
    agent_config = {"configurable": {"thread_id": thread_id}}
    # Pass the thread-id to establish memory for chatbot
    # Invoke the agent with the state
    result = agent.invoke(state, agent_config)

    # Convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        final_result = AIMessage(result["messages"][-1].content)
    return {"messages": [final_result]}


class RouterAgent:
    def __init__(self, debug=False):
        self.system_prompt = """ 
            You are a Router, that analyzes the input query and chooses 3 options:
            SMALLTALK: If the user input is small talk, like greetings and good byes.
            DATA: If the query is a data related question about identity, such question can involve reading from a source or writing to a source.
            DUPLICATE: if the query is asking for recommendation of duplicates, or possible duplicates.
            FEEDBACK: If user input is a response identify whether duplicate are true.
            END: Default, when its neither DATA or DUPLICATE or FEEDBACK.

            The output should only be just one word out of the possible 3 : SMALLTALK, DATA, DUPLICATE
            """
        self.smalltalk_prompt = """
            If the user request is small talk, like greetings and goodbyes, respond professionally.
            Mention that you will be able to answer questions about using Data_Agent, breifly explain its capabilities.
            """
        self.model = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini-2",
            api_version="2024-05-01-preview",
            model="gpt-4o",
        )
        self.debug = debug
        self.data_agent = DataAgent()
        self.duplicate_recommendation_agent = DuplicateRecommenderAgent()
        self.feedback_agent = FeedbackAgent()

        data_agent_node = functools.partial(agent_node, agent=self.data_agent)
        duplicate_recommendation_agent_node = functools.partial(
            agent_node, agent=self.duplicate_recommendation_agent
        )
        feedback_agent_node = functools.partial(agent_node, agent=self.feedback_agent)

        router_graph = StateGraph(RouterAgentState)
        router_graph.add_node("Router", self.call_llm)
        router_graph.add_node("Data_Agent", data_agent_node)
        router_graph.add_node("Duplicate_Recommender_Agent", duplicate_recommendation_agent_node)
        router_graph.add_node("Feedback_Agent", feedback_agent_node)
        router_graph.add_node("Small_Talk", self.respond_smalltalk)

        router_graph.add_conditional_edges(
            "Router",
            self.find_route,
            {"DATA": "Data_Agent","DUPLICATE":"Duplicate_Recommender_Agent", "FEEDBACK":"Feedback_Agent","SMALLTALK": "Small_Talk", "END": END},
        )

        # One way routing, not coming back to router
        router_graph.add_edge("Data_Agent", END)
        router_graph.add_edge("Duplicate_Recommender_Agent", END)
        router_graph.add_edge("Feedback_Agent", END)
        router_graph.add_edge("Small_Talk", END)

        # Set where there graph starts
        router_graph.set_entry_point("Router")
        self.router_graph = router_graph.compile()

    def call_llm(self, state: RouterAgentState):
        messages = state["messages"]
        if self.debug:
            print(f"Call LLM received {messages}")

        # If system prompt exists, add to messages in the front
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        # invoke the model with the message history
        result = self.model.invoke(messages)

        if self.debug:
            print(f"Call LLM result {result}")
        return {"messages": [result]}

    def respond_smalltalk(self, state: RouterAgentState):
        messages = state["messages"]
        if self.debug:
            print(f"Small talk received: {messages}")

        # If system prompt exists, add to messages in the front

        messages = [SystemMessage(content=self.smalltalk_prompt)] + messages

        # invoke the model with the message history
        result = self.model.invoke(messages)

        if self.debug:
            print(f"Small talk result {result}")
        return {"messages": [result]}

    def find_route(self, state: RouterAgentState):
        last_message = state["messages"][-1]
        if self.debug:
            print("Router: Last result from LLM : ", last_message)

        # Set the last message as the destination
        destination = last_message.content

        if self.debug:
            print(f"Destination chosen : {destination}")
        return destination
