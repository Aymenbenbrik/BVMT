# mcp_servers/database_server.py
# ──────────────────────────────────────────────────────────────────────
# PURPOSE:  Expose **PostgreSQL database operations** as MCP tools.
#
# WHAT IS MCP?
#   MCP (Model Context Protocol) is a standard that lets an LLM
#   (like GPT-4 / Claude) call external "tools" in a structured way.
#   Instead of the LLM writing raw SQL, it calls a tool like:
#       execute_query(sql="SELECT * FROM prices WHERE ticker='PGH'")
#   and this server handles the actual database connection safely.
#
# WHY A SEPARATE SERVER?
#   • Security — the LLM never sees the database password.
#   • Reusability — any agent can call these tools without knowing
#     how PostgreSQL connections work.
#   • Testability — you can mock this server in unit tests.
#
# TOOLS THAT WILL BE EXPOSED:
#   1. execute_query   → run a SELECT and return rows
#   2. insert_prices   → bulk-insert price rows
#   3. list_tables     → show available tables
#
# CONNECTION:
#   Uses environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER,
#   DB_PASSWORD) so credentials stay out of the source code.
#
# STATUS: Skeleton — actual implementation comes when PostgreSQL
#         is set up (Week 2).
# ──────────────────────────────────────────────────────────────────────

'''
MCP Database Server for BVMT Project
Exposes PostgreSQL tools to AI agents via the MCP protocol.


Run this file from a terminal:
    python mcp_servers/database_server.py


It will start and wait for tool calls. Keep it running while
your agents or notebooks are working.
'''


import asyncio
import psycopg2
import psycopg2.extras
import json
import os
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


load_dotenv()


DB_CONFIG = {
    'host':     os.getenv('DB_HOST', 'localhost'),
    'port':     int(os.getenv('DB_PORT', 5432)),
    'dbname':   os.getenv('DB_NAME', 'bvmt_db'),
    'user':     os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# ── The 5 tool functions (same logic as tested in Jupyter) ───────


def _postgres_query(sql, params=None):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or [])
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _get_price_history(ticker, start_date=None, end_date=None):
    conditions = ['dp.ticker = %s OR dp.isin_code = %s']
    params = [ticker, ticker]
    if start_date:
        conditions.append('dp.seance >= %s'); params.append(start_date)
    if end_date:
        conditions.append('dp.seance <= %s'); params.append(end_date)
    where = ' AND '.join(conditions)
    sql = f'''
        SELECT dp.seance, dp.isin_code, dp.ticker,
               dp.ouverture, dp.cloture, dp.plus_bas, dp.plus_haut,
               dp.quantite_negociee, dp.capitaux,
               ci.ma_20, ci.rsi_14, ci.volatility_20, ci.daily_return
        FROM daily_prices dp
        LEFT JOIN computed_indicators ci
               ON dp.seance=ci.seance AND dp.isin_code=ci.isin_code
        WHERE {where} ORDER BY dp.seance
    '''
    return _postgres_query(sql, params)


def _list_stocks():
    return _postgres_query('''
        SELECT isin_code, ticker, first_seen_date, last_seen_date, total_trading_days
        FROM company_metadata ORDER BY total_trading_days DESC
    ''')


def _postgres_insert(table, rows):
    if not rows: return {'inserted': 0}
    conn = get_conn()
    try:
        cols = list(rows[0].keys())
        sql = f'INSERT INTO {table} ({chr(44).join(cols)}) VALUES ({chr(44).join(["%s"]*len(cols))}) ON CONFLICT DO NOTHING'
        psycopg2.extras.execute_batch(conn.cursor(), sql, [tuple(r[c] for c in cols) for r in rows])
        conn.commit()
        return {'inserted': len(rows), 'table': table}
    finally:
        conn.close()


# ── MCP Server setup ─────────────────────────────────────────────


app = Server('bvmt-database-server')


@app.list_tools()
async def list_tools():
    return [
        types.Tool(name='postgres_query',
            description='Run any SELECT SQL on bvmt_db. Returns list of row dicts.',
            inputSchema={'type':'object','properties':{
                'sql':{'type':'string','description':'The SQL query to run'},
                'params':{'type':'array','description':'Optional query parameters','default':[]}
            },'required':['sql']}),
        types.Tool(name='get_price_history',
            description='Get full price + indicator history for one stock by ticker name or ISIN.',
            inputSchema={'type':'object','properties':{
                'ticker':{'type':'string','description':'Stock name e.g. AMEN BANK or ISIN TN0003400058'},
                'start_date':{'type':'string','description':'Optional start date YYYY-MM-DD'},
                'end_date':{'type':'string','description':'Optional end date YYYY-MM-DD'}
            },'required':['ticker']}),
        types.Tool(name='list_stocks',
            description='Return all stocks available in company_metadata.',
            inputSchema={'type':'object','properties':{},'required':[]}),
        types.Tool(name='postgres_insert',
            description='Insert rows into a table. rows is a list of dicts.',
            inputSchema={'type':'object','properties':{
                'table':{'type':'string'},
                'rows':{'type':'array','items':{'type':'object'}}
            },'required':['table','rows']}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == 'postgres_query':
            result = _postgres_query(arguments['sql'], arguments.get('params'))
        elif name == 'get_price_history':
            result = _get_price_history(
                arguments['ticker'],
                arguments.get('start_date'),
                arguments.get('end_date')
            )
        elif name == 'list_stocks':
            result = _list_stocks()
        elif name == 'postgres_insert':
            result = _postgres_insert(arguments['table'], arguments['rows'])
        else:
            result = {'error': f'Unknown tool: {name}'}


        return [types.TextContent(type='text', text=json.dumps(result, default=str))]


    except Exception as e:
        return [types.TextContent(type='text', text=json.dumps({'error': str(e)}))]


async def main():
    print('BVMT Database MCP Server starting...')
    print(f'Connected to: {DB_CONFIG["dbname"]} on {DB_CONFIG["host"]}')
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('BVMT Database MCP Server stopped.')
