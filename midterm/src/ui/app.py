import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))


# Create data directories if they don't exist
def setup_data_directories():
    root_dir = Path(__file__).parent.parent.parent
    data_dirs = [
        root_dir / "data" / "raw",
        root_dir / "data" / "processed",
        root_dir / "data" / "processed" / "qdrant",
    ]

    for dir_path in data_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


# Load environment variables from .env file in project root
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
logger.info(f"Loading environment from: {env_path}")
load_dotenv(env_path)

# Ensure data directories exist
setup_data_directories()

# Verify OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OpenAI API key not found")
    st.error("OpenAI API key not found. Please ensure it is set in your .env file.")
    st.stop()
else:
    logger.info("OpenAI API key loaded successfully")

from embedding.model import EmbeddingModel
from rag.chain import RAGChain
from rag.document_loader import GridCodeLoader
from rag.vectorstore import VectorStore


def initialize_rag():
    # Get absolute path to the data file
    root_dir = Path(__file__).parent.parent.parent
    data_path = root_dir / "data" / "raw" / "grid_code.pdf"

    if not data_path.exists():
        logger.error(f"PDF not found: {data_path}")
        st.error(
            f"Grid Code PDF not found at {data_path}. Please ensure the file exists."
        )
        st.stop()

    logger.info("Loading and processing documents...")
    # Load just first 5 pages for testing
    loader = GridCodeLoader(str(data_path), pages=7)
    documents = loader.load_and_split()
    logger.info(f"Split documents into {len(documents)} chunks")

    logger.info("Initializing embedding model...")
    embedding_model = EmbeddingModel()
    vectorstore = VectorStore(embedding_model)
    vectorstore = vectorstore.create_vectorstore(documents)
    logger.info("Vector store created successfully")

    logger.info("Initializing RAG chain...")
    return RAGChain(vectorstore)


class WeatherService:
    def __init__(self):
        self.base_url = "https://api.weather.gov"
        self.headers = {
            "User-Agent": "(Grid Code Assistant, contact@example.com)",
            "Accept": "application/json",
        }

    def get_coordinates_from_zip(self, zipcode):
        # Using a free geocoding service (consider adding rate limiting)
        response = requests.get(f"https://api.zippopotam.us/us/{zipcode}")
        if response.status_code == 200:
            data = response.json()
            return {
                "lat": data["places"][0]["latitude"],
                "lon": data["places"][0]["longitude"],
            }
        return None

    def get_forecast(self, zipcode):
        coords = self.get_coordinates_from_zip(zipcode)
        if not coords:
            return None

        # Get grid coordinates
        point_url = f"{self.base_url}/points/{coords['lat']},{coords['lon']}"
        response = requests.get(point_url, headers=self.headers)

        if response.status_code != 200:
            return None

        grid_data = response.json()
        forecast_url = grid_data["properties"]["forecast"]

        # Get forecast
        response = requests.get(forecast_url, headers=self.headers)
        if response.status_code == 200:
            return response.json()["properties"]["periods"]
        return None


def main():
    st.title("Grid Code Assistant")

    # Initialize weather service
    if "weather_service" not in st.session_state:
        st.session_state.weather_service = WeatherService()

    # Initialize RAG chain
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = initialize_rag()
        logger.info("RAG chain initialized successfully")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Grid Code Q&A", "Weather Information"])

    with tab1:
        question = st.text_input("Ask a question about the Grid Code:")

        if question:
            logger.info(f"Processing question: {question}")
            with st.spinner("Finding answer..."):
                response = st.session_state.rag_chain.invoke(question)
                logger.info("Generated response successfully")
                st.write(response["answer"])

    with tab2:
        st.header("Weather Information")
        zipcode = st.text_input("Enter ZIP code for weather forecast:", max_chars=5)

        if zipcode and len(zipcode) == 5:
            with st.spinner("Fetching weather data..."):
                forecast = st.session_state.weather_service.get_forecast(zipcode)

                if forecast:
                    # Display current conditions
                    current = forecast[0]
                    st.subheader("Current Conditions")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Temperature",
                            f"{current['temperature']}°{current['temperatureUnit']}",
                        )
                        st.write(
                            f"**Wind:** {current['windSpeed']} {current['windDirection']}"
                        )

                    with col2:
                        st.write(f"**Forecast:** {current['shortForecast']}")
                        st.write(f"**Details:** {current['detailedForecast']}")

                    # Display future forecast
                    st.subheader("Extended Forecast")
                    for period in forecast[1:4]:  # Next 3 periods
                        with st.expander(f"{period['name']}"):
                            st.write(
                                f"**Temperature:** {period['temperature']}°{period['temperatureUnit']}"
                            )
                            st.write(
                                f"**Wind:** {period['windSpeed']} {period['windDirection']}"
                            )
                            st.write(f"**Forecast:** {period['shortForecast']}")
                            st.write(f"**Details:** {period['detailedForecast']}")
                else:
                    st.error(
                        "Unable to fetch weather data. Please check the ZIP code and try again."
                    )


if __name__ == "__main__":
    main()
