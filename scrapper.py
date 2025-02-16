import os
import time
import json
import requests
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://papers.nips.cc"
THREAD_POOL_SIZE = 10
METADATA_FILE = "metadata.json"
DOWNLOAD_DIR = "NeurIPS_Papers"

def load_existing_metadata():
    """Load existing metadata to track already downloaded papers."""
    metadata_path = os.path.join(DOWNLOAD_DIR, METADATA_FILE)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    """Save metadata to a JSON file."""
    metadata_path = os.path.join(DOWNLOAD_DIR, METADATA_FILE)
    data = load_existing_metadata()
    data.append(metadata)
    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=4)

def is_already_downloaded(pdf_url):
    """Check if a paper has already been downloaded."""
    existing_metadata = load_existing_metadata()
    return any(entry["file_path"] and entry["url"] == pdf_url for entry in existing_metadata)

def download_pdf(pdf_url, file_path):
    """Download the PDF and save it to the respective folder."""
    print(f"Downloading {pdf_url}...")
    try:
        response = requests.get(pdf_url, timeout=60, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Saved {file_path}")
        else:
            print(f"Failed to download {pdf_url}, Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")

def process_paper(paper_page_url):
    """Process a single paper page to extract metadata and download the PDF."""
    print(f"Processing paper page: {paper_page_url}")
    try:
        response = requests.get(paper_page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find("h4").text.strip() if soup.find("h4") else "Unknown Title"
        author_section = soup.find("h4", text="Authors")
        author_tag = author_section.find_next_sibling("p")
        if author_tag and author_tag.find("i"):
            authors = author_tag.find("i").text.strip()
        year_match = re.search(r'/paper_files/paper/(\d{4})/', paper_page_url)
        year = int(year_match.group(1)) if year_match else 0
        
        pdf_link = soup.select_one("a[href$='-Paper-Conference.pdf'], a[href$='-Paper.pdf']")
        if pdf_link:
            pdf_url = BASE_URL + pdf_link.get('href')
            if is_already_downloaded(pdf_url):
                print(f"Skipping already downloaded paper: {pdf_url}")
                return
            
            print(f"Found PDF: {pdf_url}")
            year_folder = os.path.join(DOWNLOAD_DIR, str(year))
            os.makedirs(year_folder, exist_ok=True)
            file_name = pdf_url.split("/")[-1]
            file_path = os.path.join(year_folder, file_name)
            
            download_pdf(pdf_url, file_path)
            
            metadata = {
                "title": title,
                "url": paper_page_url,
                "authors": authors,
                "year": year,
                "file_path": file_path,
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_metadata(metadata)
    except Exception as e:
        print(f"Error processing {paper_page_url}: {e}")

def process_year(year_url):
    """Process a single year page by extracting paper links and downloading PDFs."""
    print(f"Processing year page: {year_url}")
    try:
        paper_links = BeautifulSoup(requests.get(year_url).text, 'html.parser').select("a[href$='.html']")
        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
            futures = [executor.submit(process_paper, BASE_URL + link.get('href')) for link in paper_links]
            for future in as_completed(futures):
                future.result()
    except Exception as e:
        print(f"Error processing year {year_url}: {e}")

def extract_year_from_url(url):
    try:
        return int(''.join(filter(str.isdigit, url)))
    except ValueError:
        return 0

def main():
    print("Starting NeurIPS paper scraper...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    try:
        year_urls = sorted(
            set([BASE_URL + link.get('href') for link in BeautifulSoup(requests.get(BASE_URL).text, 'html.parser').select("a[href^='/paper']")]),
            key=extract_year_from_url, reverse=True
        )
        latest_year_urls = year_urls[:5]
        print(f"Processing latest {len(latest_year_urls)} years: {latest_year_urls}")
        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
            futures = [executor.submit(process_year, url) for url in latest_year_urls]
            for future in as_completed(futures):
                future.result()
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
