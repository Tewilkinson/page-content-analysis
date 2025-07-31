import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -- Helper functions --
def fetch_page(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def parse_html(html):
    return BeautifulSoup(html, 'html.parser')


def extract_body(soup):
    # Try to get <article> or fallback to <body>
    return soup.find('article') or soup.find('body')


def get_paragraph_texts(body):
    ps = body.find_all('p')
    return [p.get_text(strip=True) for p in ps]


def count_sections(body):
    headings = body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    return len(headings)


def find_author(soup):
    # Check meta tags
    meta_author = (soup.find('meta', attrs={'name': 'author'}) or 
                   soup.find('meta', attrs={'property': 'article:author'}))
    if meta_author and meta_author.get('content'):
        return meta_author['content']
    # Check JSON-LD for author
    for tag in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(tag.string or '{}')
            if isinstance(data, dict) and 'author' in data:
                author = data['author']
                if isinstance(author, dict) and 'name' in author:
                    return author['name']
                if isinstance(author, str):
                    return author
        except json.JSONDecodeError:
            continue
    # Heuristic: look for rel=author
    a_author = soup.find('a', rel='author')
    return a_author.get_text(strip=True) if a_author else None


def extract_links(body):
    links = []
    # Exclude links inside <nav> or <footer> and filter only absolute URLs
    for a in body.find_all('a', href=True):
        href = a['href']
        # skip nav/footer links
        if a.find_parent(['nav', 'footer']):
            continue
        # only include HTTP/HTTPS URLs
        if href.startswith('http://') or href.startswith('https://'):
            links.append(href)
    return links


def compute_relevancy(text, title, keyword):
    docs = [keyword, title + ' ' + text]
    vectorizer = TfidfVectorizer().fit(docs)
    tfidf = vectorizer.transform(docs)
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

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

            # Extract data
            paragraphs = get_paragraph_texts(body)
            full_text = ' '.join(paragraphs)
            article_length = len(full_text.split())

            title_tag = soup.title.string if soup.title else ''
            relevancy = compute_relevancy(full_text, title_tag, keyword) if keyword else None

            section_count = count_sections(body)
            author = find_author(soup)
            links = extract_links(body)

            # Display results
            st.subheader("Results")
            st.write(f"**Article length (words):** {article_length}")
            if relevancy is not None:
                st.write(f"**Relevancy score:** {relevancy:.3f}")
            st.write(f"**Section count (headings):** {section_count}")
            st.write(f"**Author:** {author or 'Not found'}")
            st.write(f"**Links in body:** {len(links)}")

            if links:
                # Replace expander with a selectbox dropdown for URLs
                selected = st.selectbox("Select a link to view", options=links)
                st.markdown(f"[Navigate to selected link]({selected})")

        except requests.HTTPError as e:
            st.error(f"Failed to fetch page: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
