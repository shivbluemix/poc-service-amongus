# from langchain_core.tools import tool
# from langchain_openai import AzureChatOpenAI
# import os
# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from dotenv import load_dotenv
# # Load environment variables from .env file
# load_dotenv()

# # Tool annotation identifies a function as a tool automatically
# @tool
# def find_sum(x: int, y: int) -> int:
#     # The docstring comment describes the capabilities of the function
#     # It is used by the agent to discover the function's inputs, outputs and capabilities
#     """
#     This function is used to add two numbers and return their sum.
#     It takes two integers as inputs and returns an integer as output.
#     """
#     return x + y


# @tool
# def find_product(x: int, y: int) -> int:
#     """
#     This function is used to multiply two numbers and return their product.
#     It takes two integers as inputs and returns an integer as ouput.
#     """
#     return x * y


# # Setup the LLM
# model = AzureChatOpenAI(
#     azure_deployment="gpt-4o-mini-2", api_version="2024-05-01-preview", model="gpt-4o"
# )

# # Create list of tools available to the agent
# agent_tools = [find_sum, find_product]

# # System prompt
# system_prompt = SystemMessage(
#     """You are a Math genius who can solve math problems. Solve the
#     problems provided by the user, by using only tools available.
#     Do not solve the problem yourself"""
# )

# agent_graph = create_react_agent(
#     model=model, state_modifier=system_prompt, tools=agent_tools
# )

# inputs = {"messages": [("user", "what is the sum of 2 and 3 ?")]}

# result = agent_graph.invoke(inputs)

# # Get the final answer
# print(f"Agent returned : {result['messages'][-1].content} \n")

# print("Step by Step execution : ")
# for message in result["messages"]:
#     print(message.pretty_repr())
import uuid
from data_agent import DataAgent
from router_agent import RouterAgent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from IPython.display import Image
# To maintain memory, each request should be in the context of a thread.
# Each user conversation will use a separate thread ID
config = {"configurable": {"thread_id": uuid.uuid4()}}

user_inputs = [
    "Hello",
    # "I am looking for this person's identity, Shivam Gupta",
    # "I want full information about Shivam Gupta",
    "I am looking for this person's identity, i don't know the full name, but last name is Eroman",
    "can you give me Kallie Blackwood",
    "Give me two poential duplicate profile",
    "Yes they are duplicates, and i am 10/10 sure"
]
# data_agent = DataAgentClient()
# for input in user_inputs:
#     print(f"----------------------------------------\nUSER : {input}")
#     #Format the user message
#     user_message = {"messages":[HumanMessage(input)]}
#     #Get response from the agent
#     ai_response = data_agent.invoke(user_message,config=config)
#     #Print the response
#     print(f"AGENT : {ai_response['messages'][-1].content}")

router_agent = RouterAgent()
image_data = router_agent.router_graph.get_graph().draw_mermaid_png()

# Save the image to a file
# with open("router_graph.png", "wb") as f:
#     f.write(image_data)

# print("Image saved as 'router_graph.png'")

for input in user_inputs:
    print(f"----------------------------------------\nUSER : {input}")
    # Format the user message
    user_message = {"messages": [HumanMessage(input)]}
    # Get response from the agent
    ai_response = router_agent.router_graph.invoke(user_message, config=config)
    # Print the response
    print(f"\nAGENT : {ai_response['messages'][-1].content}")
