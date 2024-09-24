import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv()

#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_MODEL_NAME"] = "gpt-4"

info_agent = Agent(
    role="Information Agent",
    goal="Give compelling informaton about a certain topic",
    backstory="""
    You love to know information. People love and hate you for it. You win most of the quizzes at your local pub.
    """
)

task1 = Task(
    description="Tell me all about the blue-ringed octopus.",
    expected_output="Give me a quick summary and also give me 7 bullet points describing it.",
    agent=info_agent
)

crew = Crew(
    agents=[info_agent],
    tasks=[task1],
    verbose=True
)

result = crew.kickoff()

print(result)