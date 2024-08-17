# NIM_on_Yugipedia

NVIDIA Inference Microservice(NIM)是一套專為加速AI模型推理而設計的微服務工具。NIM提供了一個簡單且高效的方式來部署各種生成式AI模型，包括大型語言模型(LLM)、語音識別、影像處理等。它結合了NVIDIA的推理軟體(如 Triton Inference Server 和 TensorRT)，讓企業能夠在雲端或本地數據中心中快速部署和管理 AI 模型。

NIM的核心特點包括：

*	**優化推理性能**：NIM 針對每個模型和硬體環境進行優化，提供最佳的延遲和吞吐量，並且能大幅縮短部署時間，從幾週縮減至幾分鐘。
* **企業級支持**：NIM 作為 NVIDIA AI Enterprise 的一部分，具備企業級的安全性和可管理性，並且支援與現有的企業管理工具整合。
* **靈活部署**：NIM 支援在各大雲端平台和本地基礎設施上運行，適合不同規模的企業進行快速部署。

NIM通過簡化AI模型的部署過程，降低了技術複雜性，並提供了高效的推理性能，為企業應用中的AI模型提供了強大的支持。

當然，這個微服務也不一定只能運用於企業情境中。本專案藉由Langchain語法的RAG技術，從[遊戲王百科](yugipedia.com)提取訊息並做正確的資訊回覆。

***

首先先從擷取資料開始。

```py

def create_embeddings(embedding_path: str = "./embed"):

    embedding_path = "./embed"
    print(f"Storing embeddings to {embedding_path}")

    # Include weblink of Yu-Gi-Oh! details
    urls = [
         "https://yugipedia.com/wiki/Set_Card_Lists:The_Infinite_Forbidden_(TCG-EN)"
    ]

    # 使用html_document_loader对NeMo toolkit技术文档数据进行加载
    documents = []
    for url in urls:
        linked_documents = load_all_linked_documents(url, depth=2) #depth值自行修改
        documents.extend(linked_documents)

    #进行chunk分词分块处理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
    index_docs(url, text_splitter, texts, embedding_path)
    print("Generated embedding successfully")

```

如果在`urls`裡面加入更多鏈結，Retriever的知識庫涵蓋範圍就會變大，但相對的向量空間檔案也會變大，運行時間也會變長。

然而這樣做會導致問題，因為我們想擷取的資訊並不在網頁最表層。加上這個yugipedia網站自己有反爬蟲的手段，如果直接靠程式擷取資訊有機率會遇到403 Forbidden。

<img width="1440" alt="Screenshot 2024-08-17 at 7 17 43 PM" src="https://github.com/user-attachments/assets/ee5db1e0-f17c-453c-8dab-bc3713cf561f">

所以這裡的做法是調整這裡使用的BeautifulSoup語法，特別是自定義headers和depth來補足。

```py

# note: this cell is put before the above cell. 

import re
from typing import List, Union

import requests
from bs4 import BeautifulSoup

from urllib.parse import urljoin

# 解決 HTTP 403 Forbidden 错误.
# 请求 https://yugipedia.com/wiki/Set_Card_Lists:The_Infinite_Forbidden_(TCG-EN) 这个页面时，服务器拒绝了请求。
# 这可能是由于服务器设置了某种限制，阻止了程序直接访问该页面。可能的原因包括防止网络爬虫、需要特定的用户代理或其他访问限制。
# 绕过 403 Forbidden 错误：你可以尝试更改请求的头信息，模拟正常的浏览器请求，可能能够绕过服务器的限制。
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

def html_document_loader(url: Union[str, bytes]) -> str:
    """
    Description:
        Loads the HTML content of a document from a given URL and return its content.
    Args:
        url: The URL of the document.
    Returns:
        The content of the document.
    Raises:
        Exception: If there is an error while making the HTTP request.
    """

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 403:
            print(f"Access to {url} was forbidden (403). Skipping this page.")
            return ""# Return empty string to skip this page
        html_content = response.text
    
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # 创建Beautiful Soup对象用来解析html
        soup = BeautifulSoup(html_content, "html.parser")

        # 删除脚本和样式标签
        for script in soup(["script", "style"]):
            script.extract()

        # 从 HTML 文档中获取纯文本
        text = soup.get_text()

        # 去除空格换行符
        text = re.sub("\s+", " ", text).strip()

        return text

    except Exception as e:
        print(f"Exception {e} while processing document from {url}")
        return ""


def get_all_links(url: str) -> List[str]:
    """
    Description:
        Extracts all the links from a given URL's HTML content.
    Args:
        url: The URL of the document.
    Returns:
        A list of URLs found in the document.
    """

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 403:
            print(f"Access to {url} was forbidden (403). Skipping this page.")
            return []  # Return empty list to skip links
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 获取页面中的所有链接
        links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
        
        return links
    except Exception as e:
        print(f"Failed to retrieve links from {url} due to exception {e}")
        return []

def load_all_linked_documents(url: str, depth: int = 1) -> List[str]:
    """
    Description:
        Loads the HTML content of a document from a given URL and all linked documents up to a certain depth.
    Args:
        url: The URL of the document.
        depth: The depth of links to follow.
    Returns:
        A list of content from the document and all linked documents.
    """

    contents = []
    
    if depth < 1:
        return contents
    
    # Load the content of the initial document
    content = html_document_loader(url)
    if content:
        contents.append(content)
    
    # If we have depth to explore, get all linked documents
    if depth > 1:
        links = get_all_links(url)
        for link in links:
            contents.extend(load_all_linked_documents(link, depth - 1))
    
    return contents

```
