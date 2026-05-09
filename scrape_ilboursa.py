#!/usr/bin/env python3
# ============================================================
# BVMT News Scraper — ilboursa.com
# Standalone script (run from terminal, not Jupyter)
# ============================================================

import requests
from bs4 import BeautifulSoup
import psycopg2
import psycopg2.extras
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, date
import time
import os
import re
from urllib.parse import quote, urljoin

load_dotenv()

# ── Database connection helper ──────────────────────────────
def get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

print("Imports OK")

# ── Ticker mapping ──────────────────────────────────────────
ILBOURSA_TICKERS = {
    # ── Banking ──────────────────────────────────────────────
    'AMEN BANK':         'AB',         # CONFIRMED
    'BIAT':              'BIAT',       # CONFIRMED
    'ATTIJARI BANK':     'TJARI',      # CONFIRMED
    'BH':                'BH',         # CONFIRMED
    'BNA':               'BNA',
    'BT':                'BT',
    'STB':               'STB',
    'UIB':               'UIB',
    'ATB':               'ATB',
    'UBCI':              'UBCI',
    'WIFACK INT BANK':   'WIFACK',
    # ── Insurance ────────────────────────────────────────────
    'STAR':              'STAR',
    'ASTREE':            'ASTREE',
    'BH ASSURANCE':      'BHASS',
    'TUNIS RE':          'TUNIS RE',
    # ── Leasing / Finance ────────────────────────────────────
    'TUNISIE LEASING F': 'TL',
    'ATTIJARI LEASING':  'ATJL',
    'HANNIBAL LEASE':    'HNBL',
    'BEST LEASE':        'BHL',
    'MODERN LEASING':    'ML',
    'TUNISIE VALEURS':   'TV',
    'SPDIT - SICAF':     'SPDIT',
    'PLAC. TSIE-SICAF':  'PLTS',
    'TUNINVEST-SICAR':   'TINV',
    'BTE (ADP)':         'BTE',
    'SIMPAR':            'SIMPAR',
    # ── Industry / Food / Beverage ───────────────────────────
    'SFBT':              'SFBT',       # CONFIRMED
    'DELICE HOLDING':    'DLICE',
    'POULINA GP HOLDING':'PGH',        # CONFIRMED
    'ALKIMIA':           'ALKIMIA',
    'ADWYA':             'ADWYA',
    'SITS':              'SITS',
    'SOTIPAPIER':        'STPAP',
    'SOMOCER':           'SOMOCER',
    'SOTUMAG':           'SOTUMAG',
    'SOTUVER':           'SOTUVER',
    'SOTRAPIL':          'STPIL',
    'SIPHAT':            'SIPHAT',
    'UNIMED':            'UNIMED',
    'MAGASIN GENERAL':   'MG',
    'MONOPRIX':          'MONOPRIX',
    # ── Technology / Telecom ─────────────────────────────────
    'ONE TECH HOLDING':  'OTECH',
    'SOTETEL':           'SOTETEL',
    'CELLCOM':           'CELLCOM',
    'TELNET HOLDING':    'TLNET',
    'GIF-FILTER':        'GIF',
    # ── Auto / Transport ─────────────────────────────────────
    'ENNAKL AUTOMOBILES':'ENNAKL',
    'CITY CARS':         'CC',
    'EURO-CYCLES':       'ECY',
    'TUNISAIR':          'TUNAIR',
    'ARTES':             'ARTES',
    # ── Construction / Materials ─────────────────────────────
    'CIMENTS DE BIZERTE':'SCB',
    'ELBENE INDUSTRIE':  'ELBENE',
    'SITEX':             'SITEX',
    'SAH':               'SAH',
    'SIAME':             'SIAME',
    'ELECTROSTAR':       'ESTAR',
    'ASSAD':             'ASSAD',
    # ── Other ────────────────────────────────────────────────
    'ICF':               'ICF',
    'CIL':               'CIL',
    'ATL':               'ATL',
    'TPR':               'TPR',
    'SOPAT':             'SOPAT',
    'MPBS':              'MPBS',
    'UADH':              'UADH',
    'STEQ':              'STEQ',
    'AIR LIQUDE TSIE':   'AIRLT',
    'ATELIER MEUBLE INT':'AMI',
    'ESSOUKNA':          'ESSOUKNA',
}

print(f"Ticker mapping loaded: {len(ILBOURSA_TICKERS)} stocks")

# ── Browser session with headers ────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) '
        'Gecko/20100101 Firefox/124.0'
    ),
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'fr-TN,fr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer':         'https://www.ilboursa.com/',
    # Avoid brotli here because some environments return undecoded binary payloads.
    'Accept-Encoding': 'gzip, deflate',
    'Connection':      'keep-alive',
})

# Visit homepage first to get cookies
try:
    resp = SESSION.get('https://www.ilboursa.com/', timeout=15)
    print(f"Homepage: HTTP {resp.status_code}")
except Exception as e:
    print(f"Warning: could not reach homepage: {e}")

# ── Parse date helper ───────────────────────────────────────
def parse_date(date_str: str):
    """Convert '18/02/26 11:32' to a Python date object."""
    try:
        date_part = date_str.strip().split()[0]
        return datetime.strptime(date_part, '%d/%m/%y').date()
    except (ValueError, IndexError):
        try:
            date_part = date_str.strip().split()[0]
            return datetime.strptime(date_part, '%d/%m/%Y').date()
        except (ValueError, IndexError):
            return date.today()

# ── Core parser function ────────────────────────────────────
def parse_ilboursa_page(ticker: str, ilboursa_code: str, max_articles: int = 30) -> list:
    """
    Fetch and parse news articles for one stock from ilboursa.com.
    
    The HTML structure is flat:
        <span class="sp1">18/02/26 11:32</span>
        <a href="article-slug_59888">Article Title</a><br>
        <span class="sp1">12/02/26 16:49</span>
        <a href="another-slug_59754">Another Title</a><br>
    """
    url = f'https://www.ilboursa.com/marches/news_valeur?s={ilboursa_code}'
    articles = []

    try:
        resp = SESSION.get(url, timeout=15)

        # Handle 403 with one retry after a longer wait
        if resp.status_code == 403:
            print(f"  Got 403 for {ticker}, waiting 30s and retrying...")
            time.sleep(30)
            resp = SESSION.get(url, timeout=15)

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {ticker} — skipping")
            return []

        # Parse the HTML. Use content+explicit decode as a safer path than resp.text.
        html_text = resp.content.decode(resp.encoding or 'utf-8', errors='replace')
        soup = BeautifulSoup(html_text, 'html.parser')

        # Fallback: if nothing found, retry once without custom Accept-Encoding.
        # This helps when an intermediary still serves compressed content oddly.
        if not soup.find('span', class_='sp1'):
            retry_headers = dict(SESSION.headers)
            retry_headers['Accept-Encoding'] = 'gzip, deflate'
            retry_resp = SESSION.get(url, timeout=15, headers=retry_headers)
            if retry_resp.status_code == 200:
                retry_html = retry_resp.content.decode(retry_resp.encoding or 'utf-8', errors='replace')
                soup = BeautifulSoup(retry_html, 'html.parser')

        # Find all date spans
        date_spans = soup.find_all('span', class_='sp1')
        date_spans = date_spans[:max_articles]

        # For each date, find its article link
        for span in date_spans:
            date_text = span.get_text(strip=True)
            published = parse_date(date_text)

            # Walk forward through siblings to find the <a> tag
            a_tag = None
            cursor = span.next_sibling

            for _ in range(10):  # max 10 hops
                if cursor is None:
                    break

                if hasattr(cursor, 'name') and cursor.name == 'a':
                    a_tag = cursor
                    break

                cursor = cursor.next_sibling

            if a_tag is None:
                continue

            # Extract title
            title = a_tag.get_text(strip=True)

            # Skip short titles
            if not title or len(title) < 10:
                continue

            # Build full URL
            href = a_tag.get('href', '').strip()
            if not href:
                continue

            if href.startswith('http'):
                full_url = href
            else:
                safe_href = quote(href, safe='/-_.~')
                full_url = urljoin('https://www.ilboursa.com/marches/', safe_href)

            articles.append({
                'ticker':       ticker,
                'isin_code':    None,
                'title':        title,
                'content':      '',
                'source':       'ilboursa.com',
                'url':          full_url,
                'published_at': str(published),
                'language':     'fr',
                'credibility':  0.90,
            })

    except Exception as e:
        print(f"  Exception for {ticker}: {e}")

    return articles

# ── Enrich with ISIN codes ─────────────────────────────────
def enrich_with_isin(articles: list) -> list:
    """Add isin_code to each article by looking up the ticker in company_metadata."""
    if not articles:
        return articles

    tickers = list(set(a['ticker'] for a in articles))

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                'SELECT ticker, isin_code FROM company_metadata WHERE ticker = ANY(%s)',
                (tickers,)
            )
            isin_map = {row['ticker']: row['isin_code'] for row in cur.fetchall()}
    finally:
        conn.close()

    for a in articles:
        a['isin_code'] = isin_map.get(a['ticker'])

    return articles

# ── Insert articles into DB ────────────────────────────────
def insert_articles(articles: list) -> int:
    """Insert scraped articles. Returns count of NEW articles inserted."""
    if not articles:
        return 0

    conn = get_conn()
    inserted = 0

    try:
        with conn.cursor() as cur:
            for a in articles:
                try:
                    cur.execute('''
                        INSERT INTO news_articles
                            (ticker, isin_code, title, content, source,
                             url, published_at, language, credibility)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO NOTHING
                    ''', (
                        a.get('ticker'),
                        a.get('isin_code'),
                        a.get('title', ''),
                        a.get('content', ''),
                        a.get('source', ''),
                        a.get('url', ''),
                        a.get('published_at'),
                        a.get('language', 'fr'),
                        a.get('credibility', 0.90),
                    ))
                    if cur.rowcount == 1:
                        inserted += 1
                except Exception:
                    pass

        conn.commit()
    finally:
        conn.close()

    return inserted

# ── TEST with AMEN BANK ────────────────────────────────────
print("\nTesting parser with AMEN BANK...")
test_articles = parse_ilboursa_page('AMEN BANK', 'AB', max_articles=5)

print(f"Articles found: {len(test_articles)}\n")
for a in test_articles:
    print(f"  Title : {a['title']}")
    print(f"  Date  : {a['published_at']}")
    print(f"  URL   : {a['url']}")
    print()

if len(test_articles) == 0:
    print("⚠ WARNING: Parser returned 0 articles!")
    print("This may mean:")
    print("  1. The ilboursa.com website structure has changed")
    print("  2. The ticker code is wrong for AMEN BANK")
    print("  3. Network/firewall is blocking the request")
    print("\nTroubleshooting:")
    print("  - Check if you can manually visit:")
    print("    https://www.ilboursa.com/marches/news_valeur?s=AB")
    print("  - Verify the <span class='sp1'> and <a> tags still exist")
else:
    print("✓ Parser is working!\n")
    print("Next steps:")
    print("  1. Review the articles above to confirm they look correct")
    print("  2. Run: python scrape_ilboursa.py  (to scrape all stocks)")
    print("\nFull scrape will:")
    print(f"  - Scrape {len(ILBOURSA_TICKERS)} stocks")
    print(f"  - Take ~{len(ILBOURSA_TICKERS) * 2 // 60} minutes")
    print("  - Insert articles into your PostgreSQL database")
