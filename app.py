from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from router_agent import RouterAgent
from langchain_core.messages import HumanMessage
import uuid
from elastic_client import ElasticSearchClient

# Initialize FastAPI app
app = FastAPI(
    title="Multi Agent API",
    description="API to interact with MultiAgent",
    version="1.0.0",
)

# Initialize the RouterAgent
config = {"configurable": {"thread_id": uuid.uuid4()}}
router_agent = RouterAgent()
es = ElasticSearchClient()
es.create_person_identity_index()
es.load_person_identity()

# Define the request model
class UserMessage(BaseModel):
    message: str


# Define the response model
class AgentResponse(BaseModel):
    response: str


# API endpoint to interact with the agent
@app.post("/agent/respond", response_model=AgentResponse)
def respond_to_user(user_message: UserMessage):
    try:
        # Prepare the state with the user's message
        state = {"messages": [HumanMessage(content=user_message.message)]}

        # Invoke the router agent
        result = router_agent.router_graph.invoke(state,config=config)

        # Extract the agent's response
        if "messages" in result and result["messages"]:
            response_content = result["messages"][-1].content
        else:
            raise HTTPException(
                status_code=500, detail="Agent did not return a valid response"
            )

        return {"response": response_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
