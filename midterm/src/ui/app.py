import logging
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

# Load environment variables
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

from embedding.model import EmbeddingModel
from rag.chain import RAGChain
from rag.document_loader import GridCodeLoader
from rag.vectorstore import VectorStore


class WeatherTool:
    def __init__(self):
        self.base_url = "https://api.weather.gov"
        self.headers = {
            "User-Agent": "(Grid Code Assistant, contact@example.com)",
            "Accept": "application/json",
        }

    def get_coordinates_from_zip(self, zipcode):
        response = requests.get(f"https://api.zippopotam.us/us/{zipcode}")
        if response.status_code == 200:
            data = response.json()
            return {
                "lat": data["places"][0]["latitude"],
                "lon": data["places"][0]["longitude"],
            }
        return None

    def run(self, zipcode):
        coords = self.get_coordinates_from_zip(zipcode)
        if not coords:
            return "Invalid ZIP code or unable to get coordinates."

        point_url = f"{self.base_url}/points/{coords['lat']},{coords['lon']}"
        response = requests.get(point_url, headers=self.headers)

        if response.status_code != 200:
            return "Unable to fetch weather data."

        grid_data = response.json()
        forecast_url = grid_data["properties"]["forecast"]

        response = requests.get(forecast_url, headers=self.headers)
        if response.status_code == 200:
            current = response.json()["properties"]["periods"][0]
            return f"Current conditions: {current['temperature']}°{current['temperatureUnit']}, {current['shortForecast']}. {current['detailedForecast']}"
        return "Unable to fetch forecast data."


def initialize_rag():
    data_path = root_dir / "data" / "raw" / "grid_code.pdf"
    if not data_path.exists():
        raise FileNotFoundError(f"PDF not found: {data_path}")

    loader = GridCodeLoader(str(data_path), pages=17)
    documents = loader.load_and_split()

    embedding_model = EmbeddingModel()
    vectorstore = VectorStore(embedding_model)
    vectorstore = vectorstore.create_vectorstore(documents)

    return RAGChain(vectorstore)


class RAGTool:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def run(self, question: str) -> str:
        """Answer questions using the Grid Code."""
        response = self.rag_chain.invoke(question)
        return response["answer"]


class AgentState(TypedDict):
    """State definition for the agent."""

    messages: Annotated[list, add_messages]


def create_agent_workflow(rag_chain, weather_tool):
    """Create an agent that can use both RAG and weather tools."""

    # Define the tools
    tools = [
        Tool(
            name="grid_code_query",
            description="Answer questions about the Grid Code and electrical regulations",
            func=lambda q: rag_chain.invoke(q)["answer"],
        ),
        Tool(
            name="get_weather",
            description="Get weather forecast for a ZIP code. Input should be a 5-digit ZIP code.",
            func=weather_tool.run,
        ),
    ]

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Get the agent prompt
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the executor
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


def main():
    st.title("Grid Code & Weather Assistant")

    # Initialize if not in session state
    if "initialized" not in st.session_state:
        rag_chain = initialize_rag()
        weather_tool = WeatherTool()
        st.session_state.app = create_agent_workflow(rag_chain, weather_tool)
        st.session_state.initialized = True

    # Create the input box
    user_input = st.text_input("Ask about weather or the Grid Code:")

    if user_input:
        with st.spinner("Processing your request..."):
            # Invoke the agent executor
            result = st.session_state.app.invoke({"input": user_input})

            # Display the result
            st.write(result["output"])


if __name__ == "__main__":
    main()
