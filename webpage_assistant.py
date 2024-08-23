import httpx
from bs4 import BeautifulSoup
import re
import json
import os
from datetime import datetime
import hashlib

from phi.assistant import Assistant
from phi.llm.groq import Groq
from phi.llm.openai import OpenAIChat
from textwrap import dedent

from phi.tools import Toolkit
from phi.assistant import Assistant
from phi.tools.pubmed import PubmedTools

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import markdown2
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import Form
from urllib.parse import quote

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


llm_type = 'openai'  # or 'openai'
# llm_type = 'groq'
llm_id = 'gpt-4o-mini'
# llm_id = 'llama3-groq-8b-8192-tool-use-preview'
# llm_id = 'llama-3.1-70b-versatile'

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
    debug_mode=False, 
    show_tool_calls=False
)

ingredient_fetcher = Assistant(
    llm=llm_class(**llm_kwargs),
    tools=[WebPageTool()],
    name="ingredient fetcher",
    role="fetch the ingredients from a webpage",
    debug_mode=False, 
    show_tool_calls=False,
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
    {
    Please list the supplements main claims in the form of a list and limit the list to 5 things
    }

    ### Research
    {pubmed links in markdown format, e.g. [Study Title](https://pubmed.ncbi.nlm.nih.gov/XXXXX/)}

    ### [Summary]
    {give a summary of the benefits of the supplements ingredients}

    ### [Recommendation]
    {Compare the claims made on the supplements site with this data we collected from pubmed and make a recommendation for whether there is good data to support the health claims made about the supplement}

    </report_format>
    """
    ),
    output_format="string",
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
        "Return the report on the ingredients based on the information you found at pubmed in the <report_format> ",
        "Ensure that your response is a single, formatted string.",
        "For the Research section, format each PubMed link as a markdown link. Use the study title as the link text.",
        "Example: [Effects of probiotics on gut microbiota](https://pubmed.ncbi.nlm.nih.gov/12345678/)"
    ],
    show_tool_calls=False,
    debug_mode=False
)

app = FastAPI()

class SupplementRequest(BaseModel):
    url: str

def format_report_html(report: str, url: str) -> str:
    # Convert markdown to HTML
    html_content = markdown2.markdown(report)
    
    # Convert plain URLs to clickable links
    # url_pattern = r'(https?://\S+)'
    # html_content = re.sub(url_pattern, r'<a href="\1">\1</a>', html_content)
    
    # Wrap the content in a basic HTML structure with some styling
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Supplement Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .summary, .recommendation {{
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 10px;
                margin: 10px 0;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
                word-break: break-all;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            pre {{
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 10px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            form {{
                display: inline-block;
                margin-right: 10px;
            }}
            input[type="submit"] {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 5px 10px;
                cursor: pointer;
            }}
            input[type="submit"]:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <!-- NAVIGATION_PLACEHOLDER -->
        <h1>Supplement Analysis Report</h1>
        <p>Original URL: {url}</p>
        <h2>Converted HTML:</h2>
        {html_content}
        <h2>Original Markdown:</h2>
        <pre>{report}</pre>
    </body>
    </html>
    """
    return html

def save_report(url: str, html_content: str):
    # Create a directory for reports if it doesn't exist
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Create a subdirectory for the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    url_dir = os.path.join(reports_dir, url_hash)
    if not os.path.exists(url_dir):
        os.makedirs(url_dir)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.html"
    
    # Save the HTML content
    filepath = os.path.join(url_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return filepath

@app.post("/analyze_supplement")
async def analyze_supplement(request: SupplementRequest):
    try:
        report_generator = assistant.run(request.url)
        report = ''.join(list(report_generator))  # Convert generator to string
        report = report.strip()
        html_report = format_report_html(report, request.url)
        return {"report": html_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_supplement_html")
async def analyze_supplement_html(url: str = Form(...)):
    try:
        report_generator = assistant.run(url)
        report = ''.join(list(report_generator))  # Convert generator to string
        report = report.strip()
        html_report = format_report_html(report, url)
        
        # Save the report
        saved_path = save_report(url, html_report)
        
        # Redirect to the new report
        url_hash = hashlib.md5(url.encode()).hexdigest()
        report_name = os.path.basename(saved_path)
        return RedirectResponse(url=f"/view_report/{url_hash}/{report_name}", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import HTMLResponse

@app.get("/list_reports", response_class=HTMLResponse)
async def list_reports(url: str):
    url_hash = hashlib.md5(url.encode()).hexdigest()
    url_dir = os.path.join("reports", url_hash)
    
    if not os.path.exists(url_dir):
        return "<h1>No reports found for this URL</h1>"
    
    reports = [f for f in os.listdir(url_dir) if f.endswith('.html')]
    reports.sort(reverse=True)  # Sort by newest first
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reports for {url}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                padding: 20px;
                max-width: 800px;
                margin: 0 auto;
            }}
            h1 {{
                color: #333;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            li {{
                margin-bottom: 10px;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <h1>Reports for: {url}</h1>
        <ul>
    """
    
    for report in reports:
        report_date = report.replace("report_", "").replace(".html", "")
        formatted_date = datetime.strptime(report_date, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        html_content += f'<li><a href="/view_report/{url_hash}/{report}">{formatted_date}</a></li>'
    
    html_content += """
        </ul>
        <p><a href="/">Back to Home</a></p>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/view_report/{url_hash}/{report_name}")
async def view_report(url_hash: str, report_name: str):
    report_path = os.path.join("reports", url_hash, report_name)
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract the original URL from the report content
    url_match = re.search(r'Original URL: (https?://\S+)', content)
    if url_match:
        original_url = url_match.group(1)
        # Remove any HTML tags that might be present
        original_url = re.sub(r'<.*?>', '', original_url)
        # Ensure the URL is properly encoded
        original_url = quote(original_url)
    else:
        original_url = "Unknown URL"
    
    # Create navigation elements
    back_link = f'<p><a href="/list_reports?url={original_url}">Back to List of Reports</a></p>'
    rerun_form = f'''
    <form action="/analyze_supplement_html" method="post">
        <input type="hidden" name="url" value="{original_url}">
        <input type="submit" value="Rerun Report">
    </form>
    '''
    navigation = f'<div>{back_link}{rerun_form}</div>'
    
    # Replace the placeholder with the navigation elements
    modified_content = content.replace('<!-- NAVIGATION_PLACEHOLDER -->', navigation)
    
    return HTMLResponse(content=modified_content)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def post_form(request: Request, url: str = Form(...)):
    try:
        report_generator = assistant.run(url)
        report = ''.join(list(report_generator))
        report = report.strip()
        html_report = format_report_html(report, url)
        
        # Save the report
        saved_path = save_report(url, html_report)
        
        return templates.TemplateResponse("form.html", {
            "request": request, 
            "result": html_report, 
            "url": url,
            "show_rerun": True  # Add this line to indicate that we should show the rerun button
        })
    except Exception as e:
        return templates.TemplateResponse("form.html", {"request": request, "result": f"Error: {str(e)}", "url": url, "show_rerun": False})