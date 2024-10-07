import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
import openai # Assuming OpenAI is used for GPT-4o mini model integration

# Set up environment variables for OpenAI API key
os.environ["OPENAI_API_KEY"] = "enter your key here"

# Set up CSVSearchTool with your CSV file
csv_tool = CSVSearchTool(csv='./Data.csv')

# Uncomment the following line to use an example of a custom tool
# from analyst.tools.custom_tool import MyCustomTool

# Check our tools documentation for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class DataValidationCrew():
    """Data validation and reporting crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
  
  
    @agent
    def comparison_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['comparison_agent'],
            
            tools=[csv_tool],
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
    def comparison_task(self) -> Task:
        return Task(
            config=self.tasks_config['comparison_task'],
            agent=self.comparison_agent()
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
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            memory=True,
            cache=True,
            verbose=True,
            # process=Process.hierarchical, # Option to use a hierarchical process
        )

   
