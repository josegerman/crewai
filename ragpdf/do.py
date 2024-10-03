from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool


class dumbAgengs:

    def one_agent():
        pdf_tool = PDFSearchTool("ragpdf\\gpt-4-analysis.pdf")

        
