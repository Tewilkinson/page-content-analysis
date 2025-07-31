import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# -- Helper functions --
def fetch_page(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def parse_html(html):
    return BeautifulSoup(html, 'html.parser')


def extract_body(soup):
    return soup.find('article') or soup.find('body')


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
    links = []
    # Only consider links within paragraph tags that include 'https://'
    for p in body.find_all('p'):
        for a in p.find_all('a', href=True):
            href = a['href'].strip()
            # Exclude in-page anchors
            if href.startswith('#'):
                continue
            # Only include links that contain 'https://'
            if 'https://' in href:
                links.append(href)
    return links


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
    st.sidebar.markdown("Enter URLs (one per line), a keyword, and adjust weights for the overall score.")
    urls_input = st.sidebar.text_area("Article URLs (one per line):")
    keyword = st.sidebar.text_input("Keyword for relevancy scoring:")
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

    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if urls and keyword:
        results = []
        for url in urls:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            try:
                html = fetch_page(url)
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
