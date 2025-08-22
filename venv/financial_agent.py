from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# Load .env file variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY not found in environment variables.")

## Web Search Agent
websearch_agent = Agent(
    name="WebSearchAgent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

## Financial Agent
finance_agent = Agent(
    name="FinancialAgent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
    ],
    instructions=["Always use tables"],
    show_tool_calls=True,
    markdown=True,
)

## Multi-Agent
multi_agent = Agent(
    team=[websearch_agent, finance_agent],
    instructions=["Always use tables"],
    show_tool_calls=True,
    markdown=True,
)

multi_agent.print_response(
    "Summarize analyst recommendations for Apple Inc. (AAPL) and Google LLC (GOOGLE) "
    "and search the web for any recent news about these companies.",
    stream=True
)
