import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# THE FIX: We import 'LLM' from crewai, not from langchain
from crewai import Agent, Task, Crew, Process, LLM

# 1. Set up the LLM (The "Brain")
# THE FIX: We instantiate CrewAI's 'LLM' class
# We pass the model name as a string, which it understands.
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key="AIzaSyD5sSxFb1_ArKFby6k1wT5uBD1NeZTGIYA" # Put your actual key in these quotes
)

# 2. Define Your Agents (The "Crew")
# This is now correct. The Agent will receive an object
# from its own library and know what to do.

researcher = Agent(
  role='Senior AI Researcher',
  goal='Discover groundbreaking new trends in AI for 2026',
  backstory="""You are a senior research analyst at a top tech
  think tank. Your mission is to find the *next big thing*
  before anyone else. You are a master at sifting through
  noise to find signals.""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)

writer = Agent(
  role='Tech Content Strategist',
  goal='Craft a compelling blog post about AI trends',
  backstory="""You are a renowned tech writer, known for
  taking complex topics and making them exciting and
  easy to understand for a executive audience. You turn
  raw research into must-read articles.""",
  verbose=True,
  allow_delegation=True,
  llm=llm
)

# 3. Define The Tasks (The "To-Do List")
# This stays exactly the same.

task1 = Task(
  description="""Conduct a thorough analysis of emerging AI trends
  for 2026. Look for topics like agentic workflows,
  multimodal models, and decentralized AI.
  
  Your final output must be a bullet point list of 5 key trends
  with a brief, 2-sentence explanation for each.""",
  expected_output='A bulleted list of 5 key AI trends and their descriptions.',
  agent=researcher
)

task2 = Task(
  description="""Using the research findings from the researcher,
  write a 3-paragraph blog post. The post should be engaging,
  optimistic, and aimed at a business leader
  who wants to know *what* to pay attention to.
  
  Make sure to give it a catchy title.""",
  expected_output='A 3-paragraph blog post with a catchy title.',
  agent=writer
)

# 4. Form the Crew (Kick off the "Project")
# This also stays the same.

crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  process=Process.sequential # Tasks are done one after another
)

# 5. Run the Crew
print("Crew is kicking off...")
result = crew.kickoff()

print("\n\n########################")
print("## Here is your Crew's result:")
print("########################\n")
print(result)