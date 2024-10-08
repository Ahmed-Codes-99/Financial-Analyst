import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
import openai  # Assuming OpenAI is used for GPT-4 mini model integration

# Set up environment variables for OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Set up CSVSearchTool with your CSV file
csv_tool = CSVSearchTool(csv='./Data.csv')


@CrewBase
class DataValidationCrew:
    """Data validation and reporting crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def comparison_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['comparison_agent'],
            tools=[csv_tool],
            memory=True,
            verbose=True,
        )

    @agent
    def data_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_summarizer'],
            tools=[csv_tool],
            memory=False,  # No memory required for summarizer
            verbose=True,
        )

    @agent
    def data_validator(self) -> Agent:
        return Agent(
            config=self.agents_config['data_validator'],
            tools=[csv_tool],
            memory=False,  # No memory required for validator
            verbose=True,
        )

    @agent
    def executive_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['executive_agent'],
            tools=[csv_tool],
            memory=False,
            verbose=True,
        )

    @agent
    def messaging_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['messaging_agent'],
            tools=[csv_tool],
            memory=False,
            verbose=True,
        )

    @task
    def comparison_task(self) -> Task:
        return Task(
            config=self.tasks_config['comparison_task'],
            agent=self.comparison_agent()
            memory=True,
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
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
             memory=True,
    long_term_memory=EnhanceLongTermMemory(
        storage=LTMSQLiteStorage(
            db_path="/my_data_dir/my_crew1/long_term_memory_storage.db"
        )
    ),
    short_term_memory=EnhanceShortTermMemory(
        storage=CustomRAGStorage(
            crew_name="my_crew",
            storage_type="short_term",
            data_dir="//my_data_dir",
            model=embedder["model"],
            dimension=embedder["dimension"],
        ),
    ),
    entity_memory=EnhanceEntityMemory(
        storage=CustomRAGStorage(
            crew_name="my_crew",
            storage_type="entities",
            data_dir="//my_data_dir",
            model=embedder["model"],
            dimension=embedder["dimension"],
        ),
    ),
            
        )
