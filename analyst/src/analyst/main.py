from crewai import Process
from crew import DataValidationCrew  # Import your custom crew class

def main():
    # Initialize the crew
    crew_instance = DataValidationCrew().crew()

  

    # Kick off the crew process
    result = crew_instance.kickoff()

    # Display the final result or report
    print("Crew execution result:")
    print(result)

if __name__ == "__main__":
    main()
