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
    url = st.text_input("Enter article URL:")
    keyword = st.text_input("Enter keyword for relevancy scoring:")

    if url:
        try:
            html = fetch_page(url)
            soup = parse_html(html)
            body = extract_body(soup)

            # Extract metrics
            text = ' '.join(get_paragraph_texts(body))
            length = len(text.split())
            title = soup.title.string if soup.title else ''
            score = compute_relevancy(text, title, keyword) if keyword else None
            sections = count_sections(body)
            author = find_author(soup) or 'Not found'
            links = extract_links(body)

            st.subheader("Results")
            st.markdown(f"- **Word count:** {length}")
            if score is not None:
                st.markdown(f"- **Relevancy score:** {score:.3f}")
            st.markdown(f"- **Section headers:** {sections}")
            st.markdown(f"- **Author:** {author}")
            st.markdown(f"- **Links found:** {len(links)}")

            if links:
                # Display links in a table
                st.subheader("Extracted Links Table")
                df = pd.DataFrame({'URL': links})
                st.table(df)

        except requests.HTTPError as e:
            st.error(f"Failed to fetch page: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
