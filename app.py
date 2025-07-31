import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Optionally use Playwright for JS rendering
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# -- Helper functions --
def fetch_page(url, use_js=False):
    # Fetch page with optional JS rendering
    if use_js and PLAYWRIGHT_AVAILABLE:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            content = page.content()
            browser.close()
            return content
    if use_js and not PLAYWRIGHT_AVAILABLE:
        st.sidebar.warning("Playwright not installed; using static fetch instead.")
    # Static fetch
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def parse_html(html):
    return BeautifulSoup(html, 'html.parser')


def extract_body(soup):
    # Prefer <main>, then <article>, then [role="main"], then <body>
    main = soup.find('main')
    if main:
        return main
    article = soup.find('article')
    if article:
        return article
    role_main = soup.find(attrs={'role': 'main'})
    if role_main:
        return role_main
    return soup.find('body')


def get_paragraph_texts(body):
    return [p.get_text(strip=True) for p in body.find_all('p')]


def count_sections(body):
    return len(body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))


def find_author(soup):
    meta = (soup.find('meta', attrs={'name': 'author'}) or
            soup.find('meta', attrs={'property': 'article:author'}))
    if meta and meta.get('content'):
        return meta['content']
    for tag in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(tag.string or '{}')
            author = data.get('author')
            if isinstance(author, dict) and 'name' in author:
                return author['name']
            if isinstance(author, str):
                return author
        except json.JSONDecodeError:
            continue
    a = soup.find('a', rel='author')
    return a.get_text(strip=True) if a else None


def extract_links(body):
    links = set()
    # Collect unique external or file links: must contain a '.' indicating a domain or file extension
    for a in body.find_all('a', href=True):
        # Skip links inside nav or footer
        if a.find_parent(['nav', 'footer']):
            continue
        href = a['href'].strip()
        # Skip in-page anchors
        if href.startswith('#'):
            continue
        # Only include hrefs with a dot (e.g., domain.com/page or file.pdf)
        if '.' in href:
            links.add(href)
    return list(links)


def compute_relevancy(text, title, keyword):
    docs = [keyword, f"{title} {text}"]
    vect = TfidfVectorizer().fit(docs)
    tfidf = vect.transform(docs)
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

# -- Streamlit app --
def main():
    st.title("Article Analyzer")
    # Sidebar controls
    st.sidebar.header("Controls")
    st.sidebar.markdown(
        "Enter URLs (one per line), a keyword, adjust weights, and optionally enable JS rendering for dynamic pages."
    )
    urls_input = st.sidebar.text_area("Article URLs (one per line):")
    keyword = st.sidebar.text_input("Keyword for relevancy scoring:")
    use_js = st.sidebar.checkbox("Enable JavaScript rendering (slow)", value=False)
    # Weight sliders
    st.sidebar.subheader("Score Weights")
    w_relevancy = st.sidebar.slider("Relevancy weight", 0.0, 1.0, 0.4, 0.05)
    w_wordcount = st.sidebar.slider("Word Count weight", 0.0, 1.0, 0.2, 0.05)
    w_links = st.sidebar.slider("Outbound Links weight", 0.0, 1.0, 0.2, 0.05)
    w_sections = st.sidebar.slider("Sections weight", 0.0, 1.0, 0.1, 0.05)
    w_author = st.sidebar.slider("Author Present weight", 0.0, 1.0, 0.1, 0.05)
    # Normalize weight sum
    total = w_relevancy + w_wordcount + w_links + w_sections + w_author
    if total > 0:
        w_relevancy /= total
        w_wordcount /= total
        w_links /= total
        w_sections /= total
        w_author /= total

    # Run analysis button
    run_analysis = st.sidebar.button("Run Analysis")

    if run_analysis:
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        if not urls or not keyword:
            st.sidebar.error("Please provide both URLs and a keyword before running analysis.")
            return
        results = []
        for url in urls:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            try:
                html = fetch_page(url, use_js=use_js)
                soup = parse_html(html)
                body = extract_body(soup)
                text = ' '.join(get_paragraph_texts(body))
                # Metrics
                word_count = len(text.split())
                title = soup.title.string if soup.title else ''
                relevancy = compute_relevancy(text, title, keyword)
                sections = count_sections(body)
                has_author = int(bool(find_author(soup)))
                outbound_links = len(extract_links(body))
                results.append({
                    'Domain': domain,
                    'URL': url,
                    'Word Count': word_count,
                    'Relevancy': relevancy,
                    'Sections': sections,
                    'Author Present': has_author,
                    'Outbound Links': outbound_links
                })
            except Exception as e:
                results.append({'Domain': domain, 'URL': url, 'Error': str(e)})
        df = pd.DataFrame(results)
        # Compute overall score
        df_clean = df.dropna(subset=['Word Count', 'Relevancy', 'Sections', 'Author Present', 'Outbound Links'])
        if not df_clean.empty:
            df_norm = df_clean.copy()
            df_norm['Word Count'] /= df_norm['Word Count'].max()
            df_norm['Sections'] /= df_norm['Sections'].max()
            df_norm['Outbound Links'] /= df_norm['Outbound Links'].max()
            df_norm['Overall Score'] = (
                df_norm['Relevancy'] * w_relevancy +
                df_norm['Word Count'] * w_wordcount +
                df_norm['Outbound Links'] * w_links +
                df_norm['Sections'] * w_sections +
                df_norm['Author Present'] * w_author
            )
            df = df.merge(df_norm[['URL', 'Overall Score']], on='URL', how='left')
        # Display
        st.subheader("Batch Analysis Results")
        st.dataframe(df)
        if 'Overall Score' in df and df['Overall Score'].notnull().any():
            st.subheader("Overall Score by Domain")
            chart = df.dropna(subset=['Overall Score']).groupby('Domain')['Overall Score'].mean()
            st.bar_chart(chart)

if __name__ == '__main__':
    main()
