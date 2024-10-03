# =================================================
# Project: RAG PDF CREW
# Dev: JoseG
# =================================================

import os
from dotenv import load_dotenv
from crewai_tools import PDFSearchTool # <============= must come before other crewai imports
from crewai import Agent, Task, Crew, Process
from textwrap import dedent

load_dotenv()

# OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
# OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)

var1 = input("What is your question: ")

# =================================================
# Tool declaration
# =================================================
tool = PDFSearchTool(pdf='ragpdf/gpt-4-analysis.pdf')

# =================================================
# Agents
# =================================================
pdf_agent = Agent(
    role="Senior PDF Analyst",
    backstory=dedent(f"""You can find anything in a pdf.  The people need you."""),
    goal=dedent(f"""Uncover any information from pdf files exceptionally well."""),
    verbose=True,
    tools=[tool]
    #llm=OpenAIGPT4,
)

writer_agent = Agent(
    role="Writer",
    backstory=dedent(f"""All your life you have loved writing summaries."""),
    goal=dedent(f"""Take the information from the pdf agent and summarize it nicely."""),
    verbose=True
    #llm=OpenAIGPT35,
)

# =================================================
# Tasks
# =================================================
pdf_task = Task(
    description=dedent(
                f"""
            Tell me precisely what I need to know from the RAG tool.
            Use this as what I want to lookup: {var1}
    
            Make sure to be as accurate as possible. 
        """
            ),
            expected_output="Full analysis.",
            agent=pdf_agent,
)

writer_task = Task(
    description=dedent(
                f"""
            Take the input from task 1 and write a compelling narrative about it.
        """
            ),
            expected_output="Give me the title, then brief summary, then bullet points, and a TL;DR.",
            agent=writer_agent,
)

# =================================================
# Put crew together
# =================================================
custom_crew = Crew(
    agents=[pdf_agent, writer_agent],
    tasks=[pdf_task, writer_task],
    process=Process.sequential,
    verbose=True
)
# =================================================
# Run crew
# =================================================
result = custom_crew.kickoff()
print("\n\n################################################")
print("## Here is your custom crew run result:")
print("################################################\n")
print(result)

