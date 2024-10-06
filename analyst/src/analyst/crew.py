import os
import requests  # To connect to the cloud API
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
import openai  # Assuming OpenAI is used for GPT-4o mini model integration

# Set up environment variables for OpenAI API key
os.environ["OPENAI_API_KEY"] = "enter your key here"

# Set up CSVSearchTool with your CSV file
csv_tool = CSVSearchTool(csv='./Data.csv')

# Example: Cloud API endpoint for storing/retrieving data
CLOUD_API_URL = "https://api.example.com/memory"

# Function to fetch memory from cloud database
def fetch_memory(key):
    response = requests.get(f"{CLOUD_API_URL}/get", params={"key": key})
    if response.status_code == 200:
        return response.json()  # Assumes the API returns the data in JSON format
    return None

# Function to update memory in cloud database
def update_memory(key, value):
    data = {"key": key, "value": value}
    response = requests.post(f"{CLOUD_API_URL}/update", json=data)
    return response.status_code == 200

@CrewBase
class DataValidationCrew():
    """Data validation and reporting crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def comparison_agent(self) -> Agent:
        """Comparison Agent with memory enabled"""
        return Agent(
            config=self.agents_config['comparison_agent'],
            tools=[csv_tool],
            memory=True  # Enable memory
        )

    @agent
    def data_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_summarizer'],
            tools=[csv_tool],
        )

    @agent
    def data_validator(self) -> Agent:
        return Agent(
            config=self.agents_config['data_validator'],
            tools=[csv_tool],
        )

    @agent
    def executive_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['executive_agent'],
            tools=[csv_tool],
        )

    @agent
    def messaging_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['messaging_agent'],
            tools=[csv_tool],
        )

    @task
    def comparison_task(self) -> Task:
        """Task for the Comparison Agent"""
        return Task(
            config=self.tasks_config['comparison_task'],
            agent=self.comparison_agent()
        )

    @task
    def data_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_summary_task'],
            agent=self.data_summarizer()
        )

    @task
    def data_validation_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_validation_task'],
            agent=self.data_validator()
        )

    @task
    def executive_format_task(self) -> Task:
        return Task(
            config=self.tasks_config['executive_format_task'],
            agent=self.executive_agent(),
            output_file='final_report.md'
        )

    @task
    def stakeholder_messaging_task(self) -> Task:
        return Task(
            config=self.tasks_config['stakeholder_messaging_task'],
            agent=self.messaging_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Data Validation and Reporting crew"""

        memory_key = "latest_data_summary"
        existing_data = fetch_memory(memory_key)

        if existing_data:
            print(f"Data found in memory: {existing_data}")
            # Skip data summarizer and data validator tasks if data is found in memory
            return Crew(
                agents=[self.comparison_agent(), self.executive_agent(), self.messaging_agent()],
                tasks=[self.comparison_task(), self.executive_format_task(), self.stakeholder_messaging_task()],
                process=Process.sequential,
                verbose=True,
            )
        else:
            print("No existing data found in memory. Executing full task set...")
            # Run all agents and tasks if no data is found in memory
            crew = Crew(
                agents=[self.comparison_agent(), self.data_summarizer(), self.data_validator(), self.executive_agent(), self.messaging_agent()],
                tasks=[self.comparison_task(), self.data_summary_task(), self.data_validation_task(), self.executive_format_task(), self.stakeholder_messaging_task()],
                process=Process.sequential,
                verbose=True,
            )
            return crew
