'''
MCP Model Server for BVMT Project — SKELETON
This file will host ML model inference tools.
Today it returns placeholder responses.
Actual model code added in Sprint 2 (after Lightning AI training).
Run: python mcp_servers/model_server.py
'''


import asyncio, json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


app = Server('bvmt-model-server')


@app.list_tools()
async def list_tools():
    return [
        types.Tool(name='run_tft',
            description='[SPRINT 2] Run TFT price prediction for a stock.',
            inputSchema={'type':'object','properties':{'ticker':{'type':'string'}},'required':['ticker']}),
        types.Tool(name='run_xgboost',
            description='[SPRINT 3] Run XGBoost financial health scoring.',
            inputSchema={'type':'object','properties':{'ticker':{'type':'string'}},'required':['ticker']}),
        types.Tool(name='run_finbert',
            description='[SPRINT 3] Run FinBERT sentiment analysis on news text.',
            inputSchema={'type':'object','properties':{'text':{'type':'string'}},'required':['text']}),
        types.Tool(name='run_gnn',
            description='[SPRINT 4] Run GNN graph analysis on BVMT stock graph.',
            inputSchema={'type':'object','properties':{'ticker':{'type':'string'}},'required':['ticker']}),
    ]


@app.call_tool()
async def call_tool(name, arguments):
    placeholder = {
        'status': 'placeholder',
        'message': f'{name} not yet implemented. Will be added in Sprint 2.',
        'tool': name,
        'input': arguments
    }
    return [types.TextContent(type='text', text=json.dumps(placeholder))]


async def main():
    print('BVMT Model MCP Server starting (skeleton mode)...')
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
