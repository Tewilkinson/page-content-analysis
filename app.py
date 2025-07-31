import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -- Helper functions --
def fetch_page(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_html(html):
    return BeautifulSoup(html, 'html.parser')


def extract_body(soup):
    # Try to get <article> or fallback to <body>
    body = soup.find('article') or soup.find('body')
    return body


def get_paragraph_texts(body):
    ps = body.find_all('p')
    texts = [p.get_text(strip=True) for p in ps]
    return texts


def count_sections(body):
    headings = body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    return len(headings)


def find_author(soup):
    # Check meta tags
    meta_author = soup.find('meta', attrs={'name': 'author'}) or soup.find('meta', attrs={'property': 'article:author'})
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
                elif isinstance(author, str):
                    return author
        except json.JSONDecodeError:
            continue
    # Heuristic: look for rel=author
    a_author = soup.find('a', rel='author')
    if a_author:
        return a_author.get_text(strip=True)
    return None


def extract_links(body):
    links = [a['href'] for a in body.find_all('a', href=True)]
    return links


def compute_relevancy(text, title, keyword):
    docs = [keyword, title + ' ' + text]
    vectorizer = TfidfVectorizer().fit(docs)
    tfidf = vectorizer.transform(docs)
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score

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
            relevancy = None
            if keyword:
                relevancy = compute_relevancy(full_text, title_tag, keyword)

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

            if st.expander("Show extracted links"):
                for link in links:
                    st.markdown(f"- {link}")

        except requests.HTTPError as e:
            st.error(f"Failed to fetch page: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
