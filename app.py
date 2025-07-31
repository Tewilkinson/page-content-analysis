import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
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
    return len(body.find_all(['h1','h2','h3','h4','h5','h6']))


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
    for a in body.find_all('a', href=True):
        href = a['href'].strip()
        if a.find_parent(['nav', 'footer']):
            continue
        if href.startswith('http://') or href.startswith('https://'):
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
    st.markdown("Enter one or more URLs (one per line) and a keyword to batch-analyze articles.")

    url_list_input = st.text_area("Article URLs (one per line):")
    keyword = st.text_input("Keyword for relevancy scoring:")

    urls = [u.strip() for u in url_list_input.splitlines() if u.strip()]
    if urls and keyword:
        results = []
        for url in urls:
            try:
                html = fetch_page(url)
                soup = parse_html(html)
                body = extract_body(soup)
                paragraphs = get_paragraph_texts(body)
                text = ' '.join(paragraphs)
                word_count = len(text.split())
                title = soup.title.string if soup.title else ''
                relevancy = compute_relevancy(text, title, keyword)
                sections = count_sections(body)
                author = find_author(soup)
                has_author = bool(author)
                results.append({
                    'URL': url,
                    'Word Count': word_count,
                    'Relevancy': round(relevancy, 3),
                    'Sections': sections,
                    'Author Present': has_author
                })
            except Exception as e:
                results.append({
                    'URL': url,
                    'Word Count': None,
                    'Relevancy': None,
                    'Sections': None,
                    'Author Present': None,
                    'Error': str(e)
                })

        df = pd.DataFrame(results)
        st.subheader("Batch Analysis Results")
        st.dataframe(df)

if __name__ == '__main__':
    main()
