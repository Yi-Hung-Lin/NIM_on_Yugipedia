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

然而這樣做會導致問題。我們使用的這個yugipedia網站自己有反爬蟲的手段，如果直接擷取資訊會遇到403 Forbidden。

