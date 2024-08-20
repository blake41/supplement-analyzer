import httpx
from bs4 import BeautifulSoup
import re
import json

from phi.assistant import Assistant
from phi.llm.groq import Groq
from phi.llm.openai import OpenAIChat
from textwrap import dedent

from phi.tools import Toolkit
from phi.assistant import Assistant
from phi.tools.pubmed import PubmedTools

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class WebPageTool(Toolkit):
    def __init__(self):
        super().__init__(name="")
        self.register(self.read_webpage)

    def read_webpage(self, url: str) -> str:
        """Extract natural language from a website url"""
        # Fetch the webpage content
        response = httpx.get(url)
        html_content = response.text

        # Create a BeautifulSoup object
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Get text from HTML
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Remove blank lines and join the text
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return json.dumps({"content": text})


# llm_type = 'openai'  # or 'openai'
llm_type = 'groq'
# llm_id = 'gpt-4o-mini'
# llm_id = 'llama3-groq-8b-8192-tool-use-preview'
llm_id = 'llama-3.1-70b-versatile'

if llm_type == 'groq':
    llm_class = Groq
    llm_kwargs = {'model': llm_id}
elif llm_type == 'openai':
    llm_class = OpenAIChat
    llm_kwargs = {'model': llm_id}
else:
    raise ValueError("Invalid llm_type. Must be 'groq' or 'openai'.")


pubmed_assistant = Assistant(
    llm=llm_class(**llm_kwargs),
    name="pubmed assistant",
    tools=[PubmedTools()], 
    role='search pubmed for each item in a list',
    debug_mode=True, 
    show_tool_calls=True
)

ingredient_fetcher = Assistant(
    llm=llm_class(**llm_kwargs),
    tools=[WebPageTool()],
    name="ingredient fetcher",
    role="fetch the ingredients from a webpage",
    debug_mode=True, 
    show_tool_calls=True,
    expected_output="return the ingredients list as a json in the format {'ingredients': ['ingredient1', 'ingredient2']}"
)

assistant = Assistant(
    llm=llm_class(**llm_kwargs),
    name="Supplement Researcher Assistant",
    team=[ingredient_fetcher, pubmed_assistant],
    expected_output=dedent(
    """\
    Return the following report as a single string:
    <report_format>
    This supplement {supplement name} has the following ingredients
    ## {ingredients}

    ### **Overview**
    {give a brief introduction of the supplement claims}

    ### Research
    {pubmed links}

    ### [Summary]
    {give a summary of the benefits of the supplements ingredients}

    ### [Recommendation]
    {Compare the claims made on the supplements site with this data we collected from pubmed and make a recommendation for whether there is good data to support the health claims made about the supplement}

    </report_format>
    """
    ),
    description=dedent(
            """\
        You are the most advanced AI system in the world.
        You have access to a set of tools and a team of AI Assistants at your disposal.
        Your goal is to assist the user in the best way possible.\
        """
        ),
        instructions = [
            "The user is going to give you a url for a supplement.\n",
            "Fetch the page contents for the supplement and find the ingredients.\n"
            "Use the pubmed assistant to search for articles on each of the ingredients you found.\n",
            "Return the report on the ingredients based on the information you found at pubmed in the <report_format> "
        ],
    show_tool_calls=True,
    debug_mode=True
)

app = FastAPI()

class SupplementRequest(BaseModel):
    url: str

@app.post("/analyze_supplement")
async def analyze_supplement(request: SupplementRequest):
    try:
        report = assistant.run(request.url)
        # Join the report strings if it's a list
        if isinstance(report, list):
            report = ''.join(report)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Remove or comment out the CLI app line
# assistant.cli_app(markdown=True)

# Add this at the end of the file
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)