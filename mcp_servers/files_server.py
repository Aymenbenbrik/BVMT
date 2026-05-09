# mcp_servers/files_server.py
# ──────────────────────────────────────────────────────────────────────
# PURPOSE:  Expose **file system & web scraping** operations as MCP tools.
#
# WHAT THIS SERVER PROVIDES:
#   Tools that agents (or an LLM) can call to:
#   1. read_file    → read a local file (CSV, JSON, TXT)
#   2. write_file   → write/append to a local file
#   3. fetch_url    → download a web page or API response
#
# WHY SEPARATE FROM THE AGENTS?
#   Agents should express *what* they want ("give me the file for PGH")
#   not *how* to do it (open(), read(), handle encoding errors…).
#   This server owns the "how" and exposes clean functions.
#
# STATUS: Skeleton — real implementations arrive in Week 2-3.
# ──────────────────────────────────────────────────────────────────────

'''
MCP Files Server for BVMT Project
Handles: CSV reading, web scraping, file operations.
PDF tools will be added on Day 6.
Run: python mcp_servers/files_server.py
'''


import asyncio, json, os, csv
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


app = Server('bvmt-files-server')


def _read_csv(filepath: str) -> list:
    '''Read a CSV file and return list of row dicts.'''
    full_path = os.path.join(BASE_DIR, filepath) if not os.path.isabs(filepath) else filepath
    rows = []
    with open(full_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _fetch_url(url: str, extract_text: bool = True) -> dict:
    '''Fetch a URL. If extract_text=True, return clean text. Else return raw HTML.'''
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; BVMT-Bot/1.0)'}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    if extract_text:
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script','style','nav','footer']):
            tag.decompose()
        text = ' '.join(soup.get_text().split())
        return {'url': url, 'text': text[:5000], 'status': resp.status_code}
    return {'url': url, 'html': resp.text[:10000], 'status': resp.status_code}


def _list_files(folder: str) -> list:
    '''List all files in a folder inside the project.'''
    full_path = os.path.join(BASE_DIR, folder)
    if not os.path.exists(full_path):
        return []
    return [{'name': f, 'size_kb': round(os.path.getsize(os.path.join(full_path,f))/1024,1)}
            for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path,f))]


@app.list_tools()
async def list_tools():
    return [
        types.Tool(name='read_csv', description='Read a CSV file. Returns list of row dicts.',
            inputSchema={'type':'object','properties':{'filepath':{'type':'string'}},'required':['filepath']}),
        types.Tool(name='fetch_url', description='Fetch and extract text from a URL.',
            inputSchema={'type':'object','properties':{
                'url':{'type':'string'},
                'extract_text':{'type':'boolean','default':True}
            },'required':['url']}),
        types.Tool(name='list_files', description='List files in a project folder.',
            inputSchema={'type':'object','properties':{'folder':{'type':'string'}},'required':['folder']}),
    ]


@app.call_tool()
async def call_tool(name, arguments):
    try:
        if name == 'read_csv':    result = _read_csv(arguments['filepath'])
        elif name == 'fetch_url': result = _fetch_url(arguments['url'], arguments.get('extract_text', True))
        elif name == 'list_files': result = _list_files(arguments['folder'])
        else: result = {'error': f'Unknown tool: {name}'}
        return [types.TextContent(type='text', text=json.dumps(result, default=str))]
    except Exception as e:
        return [types.TextContent(type='text', text=json.dumps({'error': str(e)}))]


async def main():
    print('BVMT Files MCP Server starting...')
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
